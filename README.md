# develop/homr_backend

As the title already says this (dev) branch will replace the oemer with [homr] (https://github.com/liebharc/homr). This will greatly improve performance and quality of the OMR. 

# Performance
Inference time Segnet: 20s
Inference time Encoder CNN: 0.85s * n_staff
Inference time Encoder Transformer: 0.5 * n_staff
Inference time Decoder: 0.05 * 20 * n_staff
