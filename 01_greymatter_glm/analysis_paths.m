function P = analysis_paths(analysis)
% Helper: returns the folders and participant list for one analysis.
%
% IN   analysis                      arm1, proj, asso, wb or comm
%      subjects_stopsignal_all.txt   one participant ID per line
% OUT  P   struct: glmRoot, groupRoot, mapFile, subjects

% ============================== CONFIG ==============================
projRoot     = getenv('PROJECT_ROOT');
scriptDir    = fileparts(mfilename('fullpath'));
subjectFile  = fullfile(scriptDir, 'subjects_stopsignal_all.txt');
FUNTOME_SETS = {'proj', 'asso', 'wb', 'comm'};
% ====================================================================

if nargin < 1 || ~ischar(analysis)
    error('analysis_paths: give one of arm1, %s', strjoin(FUNTOME_SETS, ', '));
end

switch analysis
    case 'arm1'
        glmRoot   = fullfile(projRoot, 'stopsignal', 'glm_all');
        mapLabel  = 'all';
        groupRoot = fullfile(projRoot, 'stopsignal', 'group_all');
    case FUNTOME_SETS
        setRoot   = fullfile(projRoot, 'stopsignal', 'functionnectome', analysis);
        glmRoot   = fullfile(setRoot, 'glm');
        mapLabel  = analysis;
        groupRoot = fullfile(setRoot, 'group');
    otherwise
        error('analysis_paths: unknown analysis ''%s''; give one of arm1, %s', ...
              analysis, strjoin(FUNTOME_SETS, ', '));
end

listed = read_list(subjectFile);
if isempty(listed)
    error('analysis_paths: no subjects read from %s', subjectFile);
end

P.analysis     = analysis;
P.glmRoot      = glmRoot;
P.mapFile      = fullfile(projRoot, 'logs', ['contrast_map_' mapLabel '.csv']);
P.groupRoot    = groupRoot;
P.subjectFile  = subjectFile;
P.nListed      = numel(listed);
P.subjects     = listed;
P.glmSubjects  = listed;
end


function items = read_list(fname)
fid = fopen(fname, 'r');
if fid < 0
    error('cannot open subject list: %s', fname);
end
items = {};
while true
    ln = fgetl(fid);
    if ~ischar(ln), break; end
    ln = strtrim(ln);
    if isempty(ln) || startsWith(ln, '#'), continue; end
    items{end + 1} = ln; %#ok<AGROW>
end
fclose(fid);
end
