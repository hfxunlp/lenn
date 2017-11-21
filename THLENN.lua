local ffi = require 'ffi'

local THLENN = {}


local generic_THLENN_h = require 'lenn.THLENN_h'
-- strip all lines starting with #
-- to remove preprocessor directives originally present
-- in THLENN.h
generic_THLENN_h = generic_THLENN_h:gsub("\n#[^\n]*", "")
generic_THLENN_h = generic_THLENN_h:gsub("^#[^\n]*\n", "")

-- THGenerator struct declaration copied from torch7/lib/TH/THRandom.h
local base_declarations = [[
typedef void THLENNState;

typedef struct {
  unsigned long the_initial_seed;
  int left;
  int seeded;
  unsigned long next;
  unsigned long state[624]; /* the array for the state vector 624 = _MERSENNE_STATE_N  */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid;
} THGenerator;
]]

-- polyfill for LUA 5.1
if not package.searchpath then
   local sep = package.config:sub(1,1)
   function package.searchpath(mod, path)
      mod = mod:gsub('%.', sep)
      for m in path:gmatch('[^;]+') do
         local nm = m:gsub('?', mod)
         local f = io.open(nm, 'r')
         if f then
            f:close()
            return nm
         end
     end
   end
end

-- load libTHLENN
THLENN.C = ffi.load(package.searchpath('libTHLENN', package.cpath))

ffi.cdef(base_declarations)

-- expand macros, allow to use original lines from lib/THLENN/generic/THLENN.h
local preprocessed = string.gsub(generic_THLENN_h, 'TH_API void THLENN_%(([%a%d_]+)%)', 'void THLENN_TYPE%1')

local replacements =
{
   {
      ['TYPE'] = 'Double',
      ['accreal'] = 'double',
      ['THTensor'] = 'THDoubleTensor',
      ['THIndexTensor'] = 'THLongTensor',
      ['THIntegerTensor'] = 'THIntTensor',
      ['THIndex_t'] = 'long',
      ['THInteger_t'] = 'int'
   },
   {
      ['TYPE'] = 'Float',
      ['accreal'] = 'double',
      ['THTensor'] = 'THFloatTensor',
      ['THIndexTensor'] = 'THLongTensor',
      ['THIntegerTensor'] = 'THIntTensor',
      ['THIndex_t'] = 'long',
      ['THInteger_t'] = 'int'
    }
}

for i=1,#replacements do
   local r = replacements[i]
   local s = preprocessed
   for k,v in pairs(r) do
      s = string.gsub(s, k, v)
   end
   ffi.cdef(s)
end

THLENN.NULL = ffi.NULL or nil

function THLENN.getState()
   return ffi.NULL or nil
end

function THLENN.optionalTensor(t)
   return t and t:cdata() or THLENN.NULL
end

local function extract_function_names(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THLENN_%(([%a%d_]+)%)') do
      t[#t+1] = n
   end
   return t
end

function THLENN.bind(lib, base_names, type_name, state_getter)
   local ftable = {}
   local prefix = 'THLENN_' .. type_name
   for i,n in ipairs(base_names) do
      -- use pcall since some libs might not support all functions (e.g. cunn)
      local ok,v = pcall(function() return lib[prefix .. n] end)
      if ok then
         ftable[n] = function(...) v(state_getter(), ...) end   -- implicitely add state
      else
         print('not found: ' .. prefix .. n .. v)
      end
   end
   return ftable
end

-- build function table
local function_names = extract_function_names(generic_THLENN_h)

THLENN.kernels = {}
THLENN.kernels['torch.FloatTensor'] = THLENN.bind(THLENN.C, function_names, 'Float', THLENN.getState)
THLENN.kernels['torch.DoubleTensor'] = THLENN.bind(THLENN.C, function_names, 'Double', THLENN.getState)

torch.getmetatable('torch.FloatTensor').THLENN = THLENN.kernels['torch.FloatTensor']
torch.getmetatable('torch.DoubleTensor').THLENN = THLENN.kernels['torch.DoubleTensor']

function THLENN.runKernel(f, type, ...)
   local ftable = THLENN.kernels[type]
   if not ftable then
      error('Unsupported tensor type: '..type)
   end
   local f = ftable[f]
   if not f then
      error(string.format("Function '%s' not found for tensor type '%s'.", f, type))
   end
   f(...)
end

return THLENN
