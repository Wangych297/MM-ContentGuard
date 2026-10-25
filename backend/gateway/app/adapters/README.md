# Adapters

每个 adapter 负责调用一个底层审核服务，并把服务返回转换成统一审核结果：

```text
text_adapter.py
image_adapter.py
video_adapter.py
```

统一输出后再由 gateway 决策是否允许消息发出。
