package com.aicelen.andromr;

import java.nio.FloatBuffer;
import java.util.HashMap;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;

public class Decoder {
    private OrtEnvironment env;
    private OrtSession session;
    private static final int NUM_CACHE_LAYERS = 32;
    private static final int CACHE_HEADS = 8;
    private static final int CACHE_DIM = 64;
    private static final int MAX_SEQ_LEN = 608;

    public void load(String Path) throws OrtException{
        this.env = OrtEnvironment.getEnvironment();

        SessionOptions options = new SessionOptions();

        this.session = env.createSession(Path, options);    
    }

    public long[][] generate(FloatBuffer context_fb) throws OrtException{
        long out_rhythm = 1;
        long out_pitch = 0;
        long out_lift = 0;
        long out_articulations = 0;
        long out_pos = 0;

        OnnxTensor context = OnnxTensor.createTensor(env, context_fb, new long[]{1, 1280, 512});

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
        long[][] token_output = new long[5][MAX_SEQ_LEN];

        // Inference Loop
        for (int step = 0; step < MAX_SEQ_LEN; step++) {
            long[][] x_rhythm = {{out_rhythm}};
            long[][] x_lift = {{out_lift}};
            long[][] x_pitch = {{out_pitch}};
            long[][] x_articulations ={{out_articulations}};

            HashMap<String, OnnxTensor> inputs = new HashMap<String, OnnxTensor>();

            if (step == 0){
                inputs.put("context", context);
            }
            else {
                inputs.put("context", OnnxTensor.createTensor(
                    this.env, 
                    FloatBuffer.wrap(new float[0]), 
                    new long[]{1, 0, 512}
                ));
            }
            inputs.put("lifts", OnnxTensor.createTensor(this.env, x_lift));
            inputs.put("pitchs", OnnxTensor.createTensor(this.env, x_pitch));
            inputs.put("rhythms", OnnxTensor.createTensor(this.env, x_rhythm));
            inputs.put("articulations", OnnxTensor.createTensor(this.env, x_articulations));
            inputs.put("cache_len", OnnxTensor.createTensor(env, new long[]{step}));

            for (int i = 0; i < NUM_CACHE_LAYERS; i++) {
                    inputs.put(kvInputNames[i], cache[i]);
                }
            OrtSession.Result result = session.run(inputs);

            // Convert Results to individual OnnxTensors
            OnnxTensor rhythmsp = (OnnxTensor) result.get("out_rhythms").get();
            OnnxTensor liftsp = (OnnxTensor) result.get("out_lifts").get();
            OnnxTensor pitchsp = (OnnxTensor) result.get("out_pitchs").get();
            OnnxTensor articulationsp = (OnnxTensor) result.get("out_articulations").get();
            OnnxTensor positionsp = (OnnxTensor) result.get("out_positions").get();

            // Convert OnnxTesnor to long Values
            out_rhythm = ((long[]) rhythmsp.getValue())[0];
            out_lift = ((long[]) liftsp.getValue())[0];
            out_pitch = ((long[]) pitchsp.getValue())[0];
            out_articulations = ((long[]) articulationsp.getValue())[0];
            out_pos = ((long[]) positionsp.getValue())[0];

            // Add result to result arrays
            // We use +1 because of the model outputs 0, it would not be possible to find 
            // the end of the sequence (array is filled with 0)
            token_output[0][step] = out_rhythm + 1;
            token_output[1][step] = out_lift + 1;
            token_output[2][step] = out_pitch + 1;
            token_output[3][step] = out_articulations + 1;
            token_output[4][step] = out_pos + 1;

            if (out_rhythm == 2) {
                break;
            }
            System.out.println(out_rhythm);

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
        }
        return token_output;
    }
}
