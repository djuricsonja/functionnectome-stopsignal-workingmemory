function miss = group_level_stop_signal(analysis)
% Group-level one-sample t-tests over the first-level contrast images.
%
% IN   analysis   arm1, proj, asso, wb or comm
%      <glm folder>/<sub>/con_*.nii        made by make_contrast_stop_signal.m
%      logs/contrast_map_<all|set>.csv   made by make_contrast_stop_signal.m
% OUT  <group folder>/<contrast>/{SPM.mat, beta_*.nii, spmT_*.nii, mask.nii,
%      subjects_in_analysis.txt}

% ============================== CONFIG ==============================
spmPath   = getenv('SPM_PATH');

CONTRASTS = {'SuccStop_vs_Go', 'UnsuccStop_vs_Go', 'SuccStop_vs_UnsuccStop'};
% ====================================================================

if nargin < 1
    error('give the analysis: arm1, proj, asso, wb or comm');
end
addpath(fileparts(mfilename('fullpath')));
P = analysis_paths(analysis);
glmRoot   = P.glmRoot;
groupRoot = P.groupRoot;
mapFile   = P.mapFile;

map = read_map(mapFile);
inList = ismember(map.subject, P.subjects);
fprintf('analysis:     %s\n', P.analysis);
fprintf('subjects:     %d, read from %s\n', numel(P.subjects), P.subjectFile);
fprintf('glm root:     %s\n', glmRoot);
fprintf('contrast map: %s  (%d rows, %d from subjects not in this analysis, left out)\n', ...
        mapFile, numel(map.subject), nnz(~inList));
noRows = setdiff(P.subjects, map.subject);
if ~isempty(noRows)
    fprintf('   in this analysis but absent from the contrast map: %s\n', strjoin(noRows, ', '));
end
fprintf('group root:   %s\n\n', groupRoot);

addpath(spmPath);
spm('Defaults', 'fMRI');
spm_jobman('initcfg');
spm_get_defaults('cmdline', true);

miss = {};

for k = 1:numel(CONTRASTS)
    cname = CONTRASTS{k};
    sel = find(strcmp(map.contrast, cname) & inList);

    scans = cell(numel(sel), 1);
    subs = cell(numel(sel), 1);
    absent = {};
    for i = 1:numel(sel)
        s = map.subject{sel(i)};
        f = fullfile(glmRoot, s, sprintf('con_%04d.nii', map.con_number(sel(i))));
        if ~exist(f, 'file')
            absent{end + 1} = s; %#ok<AGROW>
            continue
        end
        scans{i} = [f ',1'];
        subs{i} = s;
    end
    keep = ~cellfun(@isempty, scans);
    scans = scans(keep);
    subs = subs(keep);

    fprintf('%-24s n = %d\n', cname, numel(scans));
    if ~isempty(absent)
        fprintf('   con image missing on disk for: %s\n', strjoin(absent, ', '));
    end
    if numel(scans) < 2
        warning('%s: too few images, skipping', cname);
        miss{end + 1} = cname; %#ok<AGROW>
        continue
    end

    outDir = fullfile(groupRoot, cname);
    if exist(outDir, 'dir'), rmdir(outDir, 's'); end
    mkdir(outDir);

    clear matlabbatch
    matlabbatch{1}.spm.stats.factorial_design.dir = {outDir};
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = scans;
    matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, ...
                                                           'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, ...
                                                                 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

    matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(outDir, 'SPM.mat')};
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

    matlabbatch{3}.spm.stats.con.spmmat = {fullfile(outDir, 'SPM.mat')};
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = [cname '_pos'];
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = [cname '_neg'];
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = -1;
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
    matlabbatch{3}.spm.stats.con.delete = 1;

    try
        spm_jobman('run', matlabbatch);
        write_list(fullfile(outDir, 'subjects_in_analysis.txt'), subs);
        fprintf('   -> %s\n', outDir);
    catch ME
        warning('%s failed: %s', cname, ME.message);
        miss{end + 1} = cname; %#ok<AGROW>
    end
end

fprintf('\n%s\n', repmat('=', 1, 72));
for k = 1:numel(CONTRASTS)
    d = fullfile(groupRoot, CONTRASTS{k});
    ok = exist(fullfile(d, 'SPM.mat'), 'file') == 2;
    nfile = fullfile(d, 'subjects_in_analysis.txt');
    n = 0;
    if exist(nfile, 'file'), n = numel(read_list(nfile)); end
    if ok, status = 'estimated'; else, status = 'FAILED'; end
    fprintf('  %-24s %-10s n = %d\n', CONTRASTS{k}, status, n);
end
if ~isempty(miss)
    fprintf('  FAILED: %s\n', strjoin(miss, ', '));
end
fprintf('%s\n', repmat('=', 1, 72));
end


function map = read_map(fname)
fid = fopen(fname, 'r');
if fid < 0, error('cannot open %s', fname); end
fgetl(fid);
map.subject = {}; map.contrast = {}; map.con_number = [];
while true
    ln = fgetl(fid);
    if ~ischar(ln), break; end
    ln = strtrim(ln);
    if isempty(ln), continue; end
    parts = strsplit(ln, ',');
    map.subject{end + 1} = parts{1};
    map.contrast{end + 1} = parts{2};
    map.con_number(end + 1) = str2double(parts{3});
end
fclose(fid);
end


function write_list(fname, items)
fid = fopen(fname, 'w');
for i = 1:numel(items), fprintf(fid, '%s\n', items{i}); end
fclose(fid);
end


function items = read_list(fname)
fid = fopen(fname, 'r');
items = {};
while true
    ln = fgetl(fid);
    if ~ischar(ln), break; end
    if ~isempty(strtrim(ln)), items{end + 1} = strtrim(ln); end %#ok<AGROW>
end
fclose(fid);
end
