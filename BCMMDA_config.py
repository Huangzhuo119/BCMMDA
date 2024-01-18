from datetime import datetime
def SSCNN_args():
    config = {
             'embed_d_size': 512,
              'embed_p_size': 512,
              'd_channel_size': [[19,512],[19,256, 512],[19,128,256, 512],[19,64,128,256, 512],[19,32,64,128,256, 512]],
              'p_channel_size': [[181,512],[19,256, 512],[19,128,256, 512],[19,64,128,256, 512],[19,32,64,128,256, 512]],
              'filter_d_size': [32,32, 32,32],
              'filter_p_size': [32,32, 64],
              'batch_size': 8,
              'epochs': 5,#原先为100
              'num_embedding': 32,#原来是32
              'dropout': 0.4 ,#原来是0.5
              'fc_size': [1024, 512, 256,],
              'lr':2e-5,#原来是1e-5
              'type':0,
              'n_classes':1,#输出图的通道数也就是最终得到几张特征图
              'clip':True,
                'stop_counter':20,
                'conv':40,
                'dim':32,
                'protein_kernel':[4, 8, 10,2],
                'drug_kernel' :[4, 6, 2,1],

                 'test_split': 0.2,
        'validation_split': 0.2,

              }



    return config
