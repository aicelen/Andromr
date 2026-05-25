package com.aicelen.andromr;

import java.nio.FloatBuffer;
import java.util.List;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.Accelerator;
import java.util.HashSet;
import com.google.ai.edge.litert.CompiledModel.Options;
import com.google.ai.edge.litert.CompiledModel.CpuOptions;
import com.google.ai.edge.litert.TensorBuffer;

public class LiteRTModel {
    private CompiledModel model;
    private List<TensorBuffer> input_buffers;
    private List<TensorBuffer> output_buffers;

    public void load(String path, int threads) throws Exception {
        HashSet<Accelerator> acc = new HashSet<Accelerator>();
        acc.add(Accelerator.CPU);
        Options opts = new Options(acc);
        CpuOptions cpu_opts = new CpuOptions(threads, null, null);
        opts.setCpuOptions(cpu_opts);
        this.model = CompiledModel.create(path, opts);
        this.input_buffers = this.model.createInputBuffers();
        this.output_buffers = this.model.createOutputBuffers();
    }

    private void writeInput(FloatBuffer image) throws Exception {
        TensorBuffer buf_in = this.input_buffers.get(0);
        float[] input = new float[image.remaining()];
        image.get(input);
        buf_in.writeFloat(input);

        this.model.run(this.input_buffers, this.output_buffers);
    }

    public long[] runInt(FloatBuffer image) throws Exception {
        writeInput(image);
        return this.output_buffers.get(0).readLong();
    }

    public float[] runFloat(FloatBuffer image) throws Exception {
        writeInput(image);
        return this.output_buffers.get(0).readFloat();
    }

    public void close() {
        if (this.model != null) {
            this.model.close();
        }
        if (this.input_buffers != null) this.input_buffers.clear();
        if (this.output_buffers != null) this.output_buffers.clear();
    }
}
