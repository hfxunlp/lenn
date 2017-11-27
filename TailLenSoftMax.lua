local TailLenSoftMax, parent = torch.class('lenn.TailLenSoftMax', 'nn.Module')

function TailLenSoftMax:__init()
   parent.__init(self)
   self.gradInput = {torch.Tensor()}
end

function TailLenSoftMax:updateOutput(input)
   local _input, _len = unpack(input)
   _input.THLENN.TailLenSoftMax_updateOutput(
      _input:cdata(),
      self.output:cdata(),
      _len:cdata()
   )
   return self.output
end

function TailLenSoftMax:updateGradInput(input, gradOutput)
   local _input, _len = unpack(input)
   _input.THLENN.TailLenSoftMax_updateGradInput(
      _input:cdata(),
      gradOutput:cdata(),
      self.gradInput[1]:cdata(),
      self.output:cdata(),
      _len:cdata()
   )
   if not self.gradInput[2] then
      self.gradInput[2] = _len.new()
   end
   self.gradInput[2]:resizeAs(_len):zero()
   return self.gradInput
end
