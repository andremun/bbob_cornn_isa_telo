function tblout = tblvertcat(tbl)
arguments (Repeating)
    tbl table
end
%--------------------------------------------------------------------------
% Author(s): Sterling Baird
%
% Date: 2020-09-05
%
% Description: vertically catenate any number of tables with different
% variables, filling in dummy values where necessary.
%
% Inputs:
%  tbl - table, where each table can have a different number of rows and
%  same and/or different variables*
%
% Outputs:
%  tblout - vertically catenated table
%
% Usage:
%  tblout = tblvertcat(tbl1,tbl2);
%  tblout = tblvertcat(tbl1,tbl2,tbl3);
%
% Notes:
%  See https://www.mathworks.com/matlabcentral/answers/179290-merge-tables-with-different-dimensions
%  and https://www.mathworks.com/matlabcentral/answers/410053-outerjoin-tables-with-identical-variable-names-and-unique-non-unique-keys
%
%  *variables of the same name must also be of the same datatype.
%--------------------------------------------------------------------------
%% table properties
ntbls = length(tbl); %number of tables
nrowslist = cellfun(@height,tbl); %number of rows for each table
%% assign temporary ids going from 1 to total # rows among all tables
tableIDtmp = [0 cumsum(nrowslist)];
for n = 1:ntbls
    %variable names
    varnames = tbl{n}.Properties.VariableNames;
    %make sure tableID isn't already a variable name
    if any(strcmp('tableID',varnames))
        error(['tableID is a variable name for tbl{' int2str(n) '}, remove or rename this variable name.'])
    end
    %assign range
    tableID = tableIDtmp(n)+1:tableIDtmp(n+1);
    tbl{n}.tableID = tableID.';
end
%% catenate table pairs
%unpack first table
t1 = tbl{1};
for n = 2:ntbls
        % unpack next table
        t2 = tbl{n};
        
        %variable names
        t1names = t1.Properties.VariableNames;
        t2names = t2.Properties.VariableNames;
        
        %shared variable names
        sharednames = intersect(t1names,t2names);
        
        %catenation
        t1 = outerjoin(t1,tbl{n},'Key',['tableID',sharednames],'MergeKeys',true);
end
%remove temporary ids
tblout = removevars(t1,'tableID');
end
