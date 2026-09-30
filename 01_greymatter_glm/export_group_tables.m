function export_group_tables(analysis)
% Export full results tables as CSV for every group contrast and threshold.
%
% IN   analysis   arm1, proj, asso, wb or comm
%      <group folder>/<contrast>/SPM.mat   made by group_level_stop_signal.m
% OUT  <group folder>/<contrast>/tables/<contrast>_<pos|neg>_<threshold>.csv

% ============================== CONFIG ==============================
spmPath   = getenv('SPM_PATH');

CONTRASTS = {'SuccStop_vs_Go', 'UnsuccStop_vs_Go', 'SuccStop_vs_UnsuccStop'};
DIRECTION = {'pos', 'neg'};

% label, threshdesc, threshold, extent
THRESH = {
    'FWE05',   'FWE',  0.05,   0
    'p001k0',  'none', 0.001,  0
    'p001k10', 'none', 0.001, 10
};
% ====================================================================

if nargin < 1
    error('give the analysis: arm1, proj, asso, wb or comm');
end
addpath(fileparts(mfilename('fullpath')));
P = analysis_paths(analysis);
groupRoot = P.groupRoot;
if ~exist(groupRoot, 'dir')
    error('group folder not found: %s', groupRoot);
end
fprintf('analysis:     %s\n', P.analysis);
fprintf('group root:   %s\n\n', groupRoot);

addpath(spmPath);
if ~exist('spm.m', 'file')
    error('SPM12 not found at %s', spmPath);
end
spm('Defaults', 'fMRI');
spm_jobman('initcfg');
spm_get_defaults('cmdline', true);
spm_get_defaults('stats.topoFDR', 1);

for k = 1:numel(CONTRASTS)
    d = fullfile(groupRoot, CONTRASTS{k});
    spmFile = fullfile(d, 'SPM.mat');
    if ~exist(spmFile, 'file')
        fprintf('%s: no SPM.mat, skipping\n', CONTRASTS{k});
        continue
    end

    outDir = fullfile(d, 'tables');
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    for c = 1:numel(DIRECTION)
        for t = 1:size(THRESH, 1)
            label = THRESH{t, 1};
            target = fullfile(outDir, ...
                sprintf('%s_%s_%s.csv', CONTRASTS{k}, DIRECTION{c}, label));

            before = dir(fullfile(d, '*.csv'));

            clear matlabbatch
            matlabbatch{1}.spm.stats.results.spmmat = {spmFile};
            matlabbatch{1}.spm.stats.results.conspec.titlestr = '';
            matlabbatch{1}.spm.stats.results.conspec.contrasts = c;
            matlabbatch{1}.spm.stats.results.conspec.threshdesc = THRESH{t, 2};
            matlabbatch{1}.spm.stats.results.conspec.thresh = THRESH{t, 3};
            matlabbatch{1}.spm.stats.results.conspec.extent = THRESH{t, 4};
            matlabbatch{1}.spm.stats.results.conspec.conjunction = 1;
            matlabbatch{1}.spm.stats.results.conspec.mask.none = 1;
            matlabbatch{1}.spm.stats.results.units = 1;
            matlabbatch{1}.spm.stats.results.export{1}.csv = true;

            old = cd(d);
            try
                spm_jobman('run', matlabbatch);
                after = dir(fullfile(d, '*.csv'));
                new = setdiff({after.name}, {before.name});
                if isempty(new)
                    fprintf('%-24s %-4s %-8s  no table written (empty map?)\n', ...
                            CONTRASTS{k}, DIRECTION{c}, label);
                else
                    movefile(fullfile(d, new{1}), target);
                    n = count_rows(target);
                    fprintf('%-24s %-4s %-8s  %3d rows -> %s\n', ...
                            CONTRASTS{k}, DIRECTION{c}, label, n, target);
                end
            catch ME
                fprintf('%-24s %-4s %-8s  FAILED: %s\n', ...
                        CONTRASTS{k}, DIRECTION{c}, label, ME.message);
            end
            cd(old);
        end
    end
end
end


function n = count_rows(f)
fid = fopen(f, 'r');
n = 0;
while ischar(fgetl(fid)), n = n + 1; end
fclose(fid);
end
