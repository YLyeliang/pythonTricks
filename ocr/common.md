# CTC loss

## tensorflow
keras中的`padding_sequence`，将多个不同长度的文本sequences padding到统一长度。 默认情况下，
从sequence开始端padding或删除字符进行截断来使得每一个批次中的samples中的长度统一。
其方向和值分别由参数`padding='post'`和`val=-1`控制。长度由`maxlen`控制，或由批次中的最长决定。

`layers.experimental.preprocessing.StringLookup`中，可以将字符映射到整数上。 通过给定vocabulary，该方法会根据参数`mask_token='''`,`num_oov_indices`
，`oov_token=[UNK]`引入一个out-of-vocabulary，表示未知的 字符。例，vocab =['a','b'],则生成['','[UNK],'a','b'].

## pytorch
`nn.CTCLoss`:

# DBNet

