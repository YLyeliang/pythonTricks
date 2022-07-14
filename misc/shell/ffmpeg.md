- ffmpeg录屏

```shell
# -i 表示录屏的左上角位置
ffmpeg -video_size 1920x1080 -framerate 25 -f x11grab -i :0.0+1080 output.mp4

# 转换编码格式
ffmpeg -i xx.mp4 -vcodec h264 xx.mkv/avi 
```