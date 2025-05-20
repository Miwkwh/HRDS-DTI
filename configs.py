from yacs.config import CfgNode as CN

_C = CN()
_C.Data = CN()
_C.Data.Path = './datasets'
_C.Data.KFold = 5
_C.Drug = CN()
_C.Drug.Node_In_Feat = 75
_C.Drug.Padding = True
_C.Drug.Hidden_Layers = [256,256,256]
_C.Drug.Node_In_Embedding = 256
_C.Drug.Nodes = 290
_C.Drug.GCN_Activation = None


_C.Protein = CN()

_C.Protein.Num_Filters = [256 , 256, 256]
_C.Protein.Kernel_Size = [3, 6, 9]

_C.Protein.Embedding_Dim = 256
_C.Protein.Padding = True
_C.Protein.Length = 1200
_C.Protein.CNN_Length = 1185


_C.SGFF = CN()

_C.SGFF.Hidden_Dim = 256
_C.SGFF.Num_Layers = 12
_C.SGFF.Num_Heads = 8
_C.SGFF.Dropout_Rate = 0.3


_C.MLP = CN()
_C.MLP.In_Dim = 256
_C.MLP.Hidden_Dim = 512
_C.MLP.Out_Dim = 64
_C.MLP.Binary = 2


_C.Global = CN()
_C.Global.Epoch = 120
_C.Global.Batch_Size = 64
_C.Global.LR = 5e-5
_C.Global.weight_decay = 5e-5


_C.Result = CN()
_C.Result.Output_Dir = "./output"
_C.Result.Save_Model = True


def get_cfg_defaults():
    return _C.clone()
