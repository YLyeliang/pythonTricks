# 各种问题记录

### docker datetime
question: 本地和服务器docker上调用python: `datetime.datetime()`返回时间相差8小时.
Reason: 本地为CST,服务器docker时间为 UTC，相差了8个小时
