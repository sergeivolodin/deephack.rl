

def cuda_process():
    print("Importing keras...")
    import os    
    os.environ['THEANO_FLAGS'] = "device=gpu1"
    from keras.models import model_from_json
    from keras import backend as K
    K.set_image_dim_ordering('th')
    
    print("Loading GPU encoder...")
    
    with open(encoder_filename + '.txt', 'r') as model_file:
        gpu_encoder = model_from_json(json.loads(next(model_file)))

    gpu_encoder.load_weights(encoder_filename + '.h5')
    
    print("Listening")
    global cuda_q
    while True:
        i = 0
        inp_array = []
        out_name = []
        ps = []
        for i in range(num_cores):
            try:
                [sn_frame, sn_features, p] = cuda_q.get_nowait()
                fr = sa.attach(sn_frame)

                inp_array.append(fr)
                out_name.append(sn_features)

                p = decompress_pipe(p)
                ps.append(p)
            except:
                break

        l = len(inp_array)

        if l == 0:
            #print("l=0")
            continue

        try:
            r = gpu_encoder.predict(np.array(inp_array))
        except:
            r = np.zeros((l, 64))

        for i in range(l):
            ft = sa.attach(out_name[i])
            p = ps[i]
            ft[:] = r[i, :]
            try:
                p.send(' ')
            except:
                continue
        #time.sleep(0.002)
        
cuda_process_p = multiprocessing.Process(target = cuda_process, args = ())
cuda_process_p.daemon = True
cuda_process_p.start()
