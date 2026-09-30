function miss = make_contrast_stop_signal(analysis)
% Define and estimate first-level contrasts.
% Adapted from a contrast script by Bernd Weber, revised by Sebastian Markett.
%
% IN   analysis   arm1, proj, asso, wb or comm
%      <glm folder>/<sub>/SPM.mat   made by firstlevel_greymatter.m (arm1)
%                                   or firstlevel_funtome.m (sets, 05_projection)
% OUT  <glm folder>/<sub>/{con_*.nii, spmT_*.nii}
%      logs/contrast_map_<all|set>.csv

% ============================== CONFIG ==============================
spmPath      = getenv('SPM_PATH');
glmSubdir    = '';

CONTRASTS = {
    'SuccStop_vs_Go',          {'succesful_stop'},   {'go_correct'}
    'UnsuccStop_vs_Go',        {'unsuccesful_stop'}, {'go_correct'}
    'SuccStop_vs_UnsuccStop',  {'succesful_stop'},   {'unsuccesful_stop'}
};
% ====================================================================

if nargin < 1
    error('give the analysis: arm1, proj, asso, wb or comm');
end
addpath(fileparts(mfilename('fullpath')));
P = analysis_paths(analysis);
glmRoot = P.glmRoot;
mapFile = P.mapFile;
subj_ids = P.glmSubjects;
nSubjects = numel(subj_ids);
if ~exist(glmRoot, 'dir')
    error('GLM folder not found: %s', glmRoot);
end

fprintf('analysis:     %s\n', P.analysis);
fprintf('subjects:     %d with a GLM in this folder (%d listed in %s)\n', nSubjects, ...
        P.nListed, P.subjectFile);
fprintf('glm root:     %s\n', glmRoot);
fprintf('contrast map: %s\n\n', mapFile);

addpath(spmPath);
spm('Defaults', 'fMRI');
spm_jobman('initcfg');
spm_get_defaults('cmdline', true);

miss = {};
mapRows = {};
skipCount = containers.Map(CONTRASTS(:,1), num2cell(zeros(1, size(CONTRASTS,1))));

for s = 1:nSubjects
    subj_id = subj_ids{s};
    try
        subj_dir = fullfile(glmRoot, subj_id, glmSubdir);
        spmFile = fullfile(subj_dir, 'SPM.mat');
        if ~exist(spmFile, 'file')
            error('SPM.mat not found at %s', spmFile);
        end

        loaded = load(spmFile, 'SPM');
        SPM = loaded.SPM;
        cd(subj_dir);

        SPM.xCon = [];
        old = [dir(fullfile(subj_dir, 'con_*.nii')); ...
               dir(fullfile(subj_dir, 'spmT_*.nii')); ...
               dir(fullfile(subj_dir, 'ess_*.nii')); ...
               dir(fullfile(subj_dir, 'spmF_*.nii'))];
        for f = 1:numel(old)
            delete(fullfile(subj_dir, old(f).name));
        end
        nCols = size(SPM.xX.X, 2);
        built = {};

        for i = 1:size(CONTRASTS, 1)
            cname = CONTRASTS{i, 1};
            posC  = CONTRASTS{i, 2};
            negC  = CONTRASTS{i, 3};

            [cvec, ok] = build_contrast(SPM, nCols, posC, negC);
            if ~ok
                skipCount(cname) = skipCount(cname) + 1;
                continue
            end

            con = spm_FcUtil('Set', cname, 'T', 'c', cvec(:), SPM.xX.xKXs);
            if isempty(SPM.xCon)
                SPM.xCon = con;
            else
                SPM.xCon(end + 1) = con;
            end
            built{end + 1} = cname; %#ok<AGROW>
        end

        if isempty(SPM.xCon)
            error('no contrasts could be built');
        end

        spm_contrasts(SPM);
        save(spmFile, 'SPM');

        for k = 1:numel(built)
            mapRows{end + 1} = sprintf('%s,%s,%d', subj_id, built{k}, k); %#ok<AGROW>
        end
        fprintf('%s  %d/%d contrasts: %s\n', subj_id, numel(built), ...
                size(CONTRASTS, 1), strjoin(built, ', '));

    catch ME
        warning('Failed for %s: %s', subj_id, ME.message);
        miss{end + 1} = subj_id; %#ok<AGROW>
    end
end

write_map(mapFile, mapRows);

fprintf('\n%s\n', repmat('=', 1, 72));
for i = 1:size(CONTRASTS, 1)
    cname = CONTRASTS{i, 1};
    fprintf('  %-24s built for %d/%d subjects (%d skipped, condition absent)\n', ...
            cname, nSubjects - skipCount(cname) - numel(miss), nSubjects, ...
            skipCount(cname));
end
fprintf('  contrast map: %s\n', mapFile);
if ~isempty(miss)
    fprintf('  FAILED: %s\n', strjoin(miss, ', '));
end
fprintf('%s\n', repmat('=', 1, 72));
end


function [cvec, ok] = build_contrast(SPM, nCols, posC, negC)
cvec = zeros(1, nCols);
ok = true;

for j = 1:numel(posC)
    idx = find_condition(SPM, posC{j});
    if isempty(idx), ok = false; return; end
    cvec(idx) = 1 / numel(posC);
end
for j = 1:numel(negC)
    idx = find_condition(SPM, negC{j});
    if isempty(idx), ok = false; return; end
    cvec(idx) = -1 / numel(negC);
end
end


function idx = find_condition(SPM, name)
pattern = ['^Sn\(\d+\)\s+' regexptranslate('escape', name) '\*bf\(1\)$'];
idx = find(~cellfun(@isempty, regexp(SPM.xX.name, pattern, 'once')));
end


function write_map(fname, rows)
d = fileparts(fname);
if ~isempty(d) && ~exist(d, 'dir')
    mkdir(d);
end
fid = fopen(fname, 'w');
fprintf(fid, 'subject,contrast,con_number\n');
for i = 1:numel(rows)
    fprintf(fid, '%s\n', rows{i});
end
fclose(fid);
end
