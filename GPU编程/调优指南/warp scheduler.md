以4090的AD102核心为例，每个SM最多可以处理48个warp，但是架构图中只有4个warp scheduler，也就是一个warp scheduler负责issue（启动）12个warps（也就是有12个warp slots）

warp slots是用于存放等待调度执行的warp的slot，warp不止有active状态，也有deactive状态，deactive不能放到slot中。


![[Pasted image 20250322155646.png]]