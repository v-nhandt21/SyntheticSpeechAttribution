function ads = getSPCup2022Datastore(root)
% 
% Copyright 2022 The MathWorks, Inc.

ads = audioDatastore(root,"FileExtensions",".wav");
rawlabels = readtable("/home/noahdrisort/Desktop/SCCup/labels.csv");

filesADS = cellfun(@getFileName,ads.Files,'UniformOutput',false);
filesCSV = rawlabels.track;
[~,~,ind] = intersect(filesADS,filesCSV);

% Arrange labels in the order of the files in the datastore
labels = categorical(rawlabels.algorithm(ind));
labelds = arrayDatastore(labels);

ads = combine(ads,labelds);
reset(ads)
end

function name = getFileName(fullFilePath)
[~,n,ext] = fileparts(fullFilePath);
name = [n,ext];
end