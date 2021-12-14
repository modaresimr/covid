source("./R/detection_fn.R")
# install.packages("xts")

library("xts")
 
id = "A36HR6Y"
dir.hr = paste("../COVID-19-Wearables/",id,"_hr.csv",sep="")
dir.step = paste("../COVID-19-Wearables/",id,"_steps.csv",sep="")

stats.result = get.stats.fn(dir.hr, dir.step, start.day=NULL, smth.k.par = 10, rest.min.par = 10, base.num.par=28, resol.tm.par=60, test.r.fixed.par=FALSE, test.r.par=NA, res.quan.par=0.9, pval.thres.par=0.01)

res.t = stats.result$res.t
cusum.t = stats.result$test.t
cusum.t.ind = stats.result$test.t.ind
cusum.test.r = stats.result$test.r.seq
cusum.pval = stats.result$test.pval
offline.result = rhr.diff.detection.fn(id,res.t, alpha=0.05)

write.csv(offline.result, file="./result/RHRDiff_offline_detection.csv" )

cusum.alarm.result = cusum.detection.fn(id,cusum.t, cusum.t.ind, cusum.pval, pval.thres=0.01, max.hour=24, dur.hour=48)

write.csv(cusum.alarm.result, file="./result/CuSum_online_detection.csv" )

result.plot.fn(id, sym.date=NA, diag.date=NA, res.t, cusum.t, cusum.test.r, offline.result, cusum.alarm.result)

