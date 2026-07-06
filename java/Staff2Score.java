package com.aicelen.andromr;

import java.nio.FloatBuffer;
import java.util.HashMap;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import com.aicelen.andromr.LiteRTModel;

public class Staff2Score {
    private OrtEnvironment env;
    private LiteRTModel encoder;
    private OrtSession decoder;
    private static final int NUM_CACHE_LAYERS = 32;
    private static final int CACHE_HEADS = 8;
    private static final int CACHE_DIM = 64;
    private static final int MAX_SEQ_LEN = 608;

    public void load(String path_encoder, String path_decoder, String threads) throws Exception{
        // Load Encoder
        this.encoder = new LiteRTModel();
        this.encoder.load(path_encoder, Integer.parseInt(threads));

        // Load Decoder
        this.env = OrtEnvironment.getEnvironment();

        SessionOptions options = new SessionOptions();

        HashMap<String, String> xnnpack = new HashMap<String, String>();
        xnnpack.put("intra_op_num_threads", threads);
        options.addXnnpack(xnnpack);

        this.decoder = env.createSession(path_decoder, options); 
        options.close();
    }

    public long[][] run(FloatBuffer image) throws Exception{
        float[] context = this.encoder.runFloat(image);
        return decode(context);
    }

    public long[][] decode(float[] context_fb) throws OrtException{
        long out_rhythm = 1;
        long out_pitch = 0;
        long out_lift = 0;
        long out_articulations = 0;
        long out_pos = 0;
        long out_slur = 0;

        OnnxTensor context = OnnxTensor.createTensor(env, FloatBuffer.wrap(context_fb), new long[]{1, 1280, 512});
        OnnxTensor empty_context = OnnxTensor.createTensor(
                    env, 
                    FloatBuffer.wrap(new float[0]), 
                    new long[]{1, 0, 512}
                );

        String[] kvInputNames = new String[NUM_CACHE_LAYERS];
        String[] kvOutputNames = new String[NUM_CACHE_LAYERS];
        for (int i = 0; i < NUM_CACHE_LAYERS; i++) {
            kvInputNames[i]  = "cache_in"  + i;
            kvOutputNames[i] = "cache_out" + i;
        }
        
        // Create empty KV cache tensors
        OnnxTensor[] cache = new OnnxTensor[NUM_CACHE_LAYERS];
        for (int i = 0; i < NUM_CACHE_LAYERS; i++) {
            cache[i] = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(new float[0]),
                new long[]{1, CACHE_HEADS, 0, CACHE_DIM}
            ); 
        }
        
        // Create array to save symbol results
        long[][] token_output = new long[6][MAX_SEQ_LEN];
        OrtSession.Result previousResult = null;

        // Inference Loop
        for (int step = 0; step < MAX_SEQ_LEN; step++) {
            OnnxTensor x_rhythm = OnnxTensor.createTensor(this.env, new long[][] {{out_rhythm}});
            OnnxTensor x_lift = OnnxTensor.createTensor(this.env, new long[][] {{out_lift}});
            OnnxTensor x_pitch = OnnxTensor.createTensor(this.env, new long[][] {{out_pitch}});
            OnnxTensor x_articulations =OnnxTensor.createTensor(this.env, new long[][] {{out_articulations}});
            OnnxTensor x_slur = OnnxTensor.createTensor(this.env, new long[][] {{out_slur}});
            OnnxTensor x_step = OnnxTensor.createTensor(env, new long[]{step});
            HashMap<String, OnnxTensor> inputs = new HashMap<String, OnnxTensor>();

            if (step == 0){
                inputs.put("context", context);
            }
            else {
                inputs.put("context", empty_context);
            }
            inputs.put("lifts", x_lift);
            inputs.put("pitchs", x_pitch);
            inputs.put("rhythms", x_rhythm);
            inputs.put("articulations", x_articulations);
            inputs.put("slurs", x_slur);
            inputs.put("cache_len", x_step);

            for (int i = 0; i < NUM_CACHE_LAYERS; i++) {
                    inputs.put(kvInputNames[i], cache[i]);
                }
            OrtSession.Result result = decoder.run(inputs);

            // Free the memory of the previous result
            // This is done now because the kv cache input
            // uses the tensors within the result

            if (previousResult != null) {
                previousResult.close();
                previousResult = null;
                for (int i = 0; i < NUM_CACHE_LAYERS; i++) {
                    cache[i] = null;
                }
            }

            // Convert Results to individual OnnxTensors
            OnnxTensor rhythmsp = (OnnxTensor) result.get("out_rhythms").get();
            OnnxTensor liftsp = (OnnxTensor) result.get("out_lifts").get();
            OnnxTensor pitchsp = (OnnxTensor) result.get("out_pitchs").get();
            OnnxTensor articulationsp = (OnnxTensor) result.get("out_articulations").get();
            OnnxTensor positionsp = (OnnxTensor) result.get("out_positions").get();
            OnnxTensor slursp = (OnnxTensor) result.get("out_slurs").get();

            // Convert OnnxTesnor to long Values
            out_rhythm = ((long[]) rhythmsp.getValue())[0];
            out_lift = ((long[]) liftsp.getValue())[0];
            out_pitch = ((long[]) pitchsp.getValue())[0];
            out_articulations = ((long[]) articulationsp.getValue())[0];
            out_pos = ((long[]) positionsp.getValue())[0];
            out_slur = ((long[]) slursp.getValue())[0];

            // Add result to result arrays
            // We use +1 because of the model outputs 0, it would not be possible to find 
            // the end of the sequence (array is filled with 0)
            token_output[0][step] = out_rhythm + 1;
            token_output[1][step] = out_lift + 1;
            token_output[2][step] = out_pitch + 1;
            token_output[3][step] = out_articulations + 1;
            token_output[4][step] = out_pos + 1;
            token_output[5][step] = out_slur + 1;

            // Close tensors
            x_lift.close();
            x_rhythm.close();
            x_articulations.close();
            x_pitch.close();
            x_slur.close();
            x_step.close();

            // EOS Token
            if (out_rhythm == 2) {
                result.close();
                break;
            }

            // Update KV Cache
            for (int i = 0; i < NUM_CACHE_LAYERS; i++) {
                // Close the previous step's tensor memory allocation
                if (cache[i] != null) {
                    cache[i].close(); 
                }
                // Grab the newly accumulated cache from this step's output
                OnnxTensor newCacheLayer = (OnnxTensor) result.get(kvOutputNames[i]).get();
                cache[i] = newCacheLayer; 
            }
            previousResult = result;
        }

        // After finishing we can close all tensors
        context.close();
        empty_context.close();

        if (previousResult != null) {
            previousResult.close();
        }
        else {
            for (OnnxTensor c : cache) {
                if (c != null) c.close();
            }
        }
        return token_output;
    }

    public void unload_model() throws OrtException {
        this.encoder.close();
        this.decoder.close();
    }
}
