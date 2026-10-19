# 形态规则历史回归样本

`yxt_002727_20261014.csv`为一心堂002727截至2026-10-14的240根腾讯前复权日K，来源为2026-10-15开盘前排查保留的快照。对应接口为`https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get`，参数`param=sz002727,day,,2026-10-14,240,qfq`；仅接受`qfqday`。

字段为date、open、high、low、close、volume；成交量已由原始手数乘100转换为股。CSV固定保存供回归核对，不作为实时行情兜底；测试与复算按目标日先截断，不引用后续K线。未在检测器中写入证券代码或名称条件。

`volume_breakout_cases.json`保存12/13/14三个不同目标日的期望。它验证规则识别和排除理由，不是盈利预测或收益回测。该样本是长期测试输入，非一次性PR截图或验收资产。
