# CTC loss

## tensorflow
keras中的`padding_sequence`，将多个不同长度的文本sequences padding到统一长度。 默认情况下，
从sequence开始端padding或删除字符进行截断来使得每一个批次中的samples中的长度统一。
其方向和值分别由参数`padding='post'`和`val=-1`控制。长度由`maxlen`控制，或由批次中的最长决定。

`layers.experimental.preprocessing.StringLookup`中，可以将字符映射到整数上。 通过给定vocabulary，该方法会根据参数`mask_token='''`,`num_oov_indices`
，`oov_token=[UNK]`引入一个out-of-vocabulary，表示未知的 字符。例，vocab =['a','b'],则生成['','[UNK],'a','b'].



在计算CTC Loss时，以TensorFlow中 `keras.backend.ctc_batch_cost`为例，模型的预测尺寸`y_pred`为`[batch,time_steps, num_categories]`,标签尺寸`y_true`为`[batch, max_string_length]`,预测长度`input_length`为`[batch,1]`,其中每个值表示每个样本的序列长度。 标签长度`label_length`为`[batch,1]`，其中每个值表示每个标签字符的实际长度。 **Note: input_length> label_length**



## pytorch
`nn.CTCLoss`:

# DBNet



## 识别数据生成

生成流程：

- 计算边缘
- 生成文本图片（包括文本img和文本mask img），随机旋转一定角度。TODO：文本颜色为随机，需要跟后续背景图片的整体颜色做一个判断
- 扭曲图像
- 缩放到期望大小(垂直文本、水平文本),缩放时根据文本距离边缘的宽度来计算文本的实际高度，并根据实际高度来缩放宽度。
- 生成背景图片，高为指定size(32),宽为文本宽度+距边缘距离
- 放置文字并对齐
- 图像增强(模糊)
- 图像名字生成、保存



在python的chr()中，中文字符为`chr(19968-40908)`

