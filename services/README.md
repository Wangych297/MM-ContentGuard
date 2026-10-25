# Services

这里放三个审核服务的代码。它们已经迁入单仓库，不需要再 clone 子仓库：

```text
services/text/   文字审核服务
services/image/  图片审核服务
services/video/  视频审核服务
```

整合目标是保留三个服务相对独立的启动方式，由 `backend/gateway` 统一调用它们。
