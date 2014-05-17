%%comments more 
% Extract the numerical values from "s" into the column vector "v". The
% variable "s" can be of any type, including struct and cell array.
% Non-numerical elements are ignored. See also the reverse rewrap.m. 


function v = unwrap(s)

v = [];   
if isnumeric(s)
  v = s(:);                        % numeric values are recast to column vector
elseif isstruct(s)    %return 1 if the expression is a structure
  v = unwrap(struct2cell(orderfields(s))); % alphabetize, conv to cell, recurse
%struct2cell -- create a new cell array from the objects stored in the struct object
elseif iscell(s)  % cell array -- N-dimensional array 
  for i = 1:numel(s)             % cell array elements are handled sequentially
    v = [v; unwrap(s{i})];
  end
end                                                   % other types are ignored
