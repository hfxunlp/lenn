require "nn"
require "lenn"
tmodstd=nn.SoftMax()
tmod=lenn.LenSoftMax()
minbsize=20
maxbsize=100
minlen=16
maxlen=128
minpadlen=4
maxpadlen=16
psg=true
firstcycle=100
for t=1, firstcycle do
	if psg then
		bsize=math.random(minbsize, maxbsize)
		lens=math.random(minlen, maxlen)
		plens=math.random(minpadlen, maxpadlen)
		lvec=torch.LongTensor(bsize):fill(lens)
		stdi=torch.randn(bsize, lens)
		i=torch.cat(stdi, torch.randn(bsize, plens))
		stdgo=torch.randn(bsize, lens)
		go=torch.cat(stdgo, torch.randn(bsize, plens))
		stdo=tmodstd:forward(stdi)
		o=tmod:forward({i, lvec})
		if not (o:narrow(2, 1, lens):equal(stdo) and o:narrow(2, lens+1, plens):equal(torch.zeros(bsize, plens)) ) then
			psg=false
			print("forward error")
		end
		stdgi=tmodstd:backward(stdi, stdgo)
		gi=tmod:backward({i, lvec}, go)[1]
		if not (gi:narrow(2, 1, lens):equal(stdgi) and gi:narrow(2, lens+1, plens):equal(torch.zeros(bsize, plens)) ) then
			psg=false
			print("backward error")
		end
	end
	xlua.progress(t, firstcycle)
end
if psg then
	print("test pass")
end
