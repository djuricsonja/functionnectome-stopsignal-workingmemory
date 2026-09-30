% First-level GLM (SPM12), masked by each subject's own brain mask.
% Adapted from a first-level script by Sebastian Markett.
%
% IN   stopsignal/staged/<sub>/sess1/func.nii        <sub>_task-stopsignal_acq-seq_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz, decompressed
%      stopsignal/staged/<sub>/sess1/sess1.mat          made by extract_onsets.py
%      stopsignal/staged/<sub>/sess1/motion_sess1.txt   made by extract_motion_parameters.py
%      stopsignal/masks/<sub>_brain_mask.nii          <sub>_task-stopsignal_acq-seq_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz, decompressed
%      subjects_stopsignal_all.txt                     one participant ID per line
% OUT  stopsignal/glm_all/<sub>/{SPM.mat, beta_*.nii, ResMS.nii, mask.nii}

% ============================== CONFIG ==============================
spmPath     = getenv('SPM_PATH');
scriptDir   = fileparts(mfilename('fullpath'));
if ~exist('subjectFile', 'var')
    subjectFile = fullfile(scriptDir, 'subjects_stopsignal_all.txt');
end
stagedRoot  = fullfile(getenv('PROJECT_ROOT'), 'stopsignal', 'staged');
maskRoot    = fullfile(getenv('PROJECT_ROOT'), 'stopsignal', 'masks');
glmRoot     = fullfile(getenv('PROJECT_ROOT'), 'stopsignal', 'glm_all');
sessions    = {'sess1'};
TR          = 2.0;
% ====================================================================

addpath(spmPath);
addpath(scriptDir);
if ~exist('spm.m', 'file')
    error('SPM12 not found at %s', spmPath);
end
spm('Defaults', 'fMRI');
spm_jobman('initcfg');
spm_get_defaults('cmdline', true);

fid = fopen(subjectFile, 'r');
if fid < 0
    error('cannot open subject list: %s', subjectFile);
end
subj_ids = {};
while true
    ln = fgetl(fid);
    if ~ischar(ln), break; end
    ln = strtrim(ln);
    if isempty(ln) || ln(1) == '#', continue; end
    subj_ids{end + 1} = ln; %#ok<SAGROW>
end
fclose(fid);

nSub = numel(subj_ids);
fprintf('subjects: %d (from %s)\n', nSub, subjectFile);
fprintf('staged:   %s\nmasks:    %s\nglm out:  %s\nTR:       %.2f s\n\n', ...
        stagedRoot, maskRoot, glmRoot, TR);

done = 0;
failed = {};
shapes = {};

for subj = 1:nSub
    subjID = subj_ids{subj};
    try
        sessDirs = cell(1, numel(sessions));
        for s = 1:numel(sessions)
            sessDir = fullfile(stagedRoot, subjID, sessions{s});
            matFile = fullfile(sessDir, [sessions{s}, '.mat']);
            if ~exist(matFile, 'file')
                error('onsets file missing: %s', matFile);
            end
            L = load(matFile, 'onsets', 'durations', 'names');

            if s == 1
                nCond = numel(L.names);
                ons = cell(nCond, numel(sessions));
                dur = cell(nCond, numel(sessions));
                ons_name = cell(nCond, 1);
            elseif numel(L.names) ~= nCond
                error('session %d has %d conditions, session 1 had %d', ...
                      s, numel(L.names), nCond);
            end

            for c = 1:nCond
                ons{c, s} = L.onsets{c};
                dur{c, s} = L.durations{c};
                if s == 1
                    ons_name{c} = L.names{c};
                end
            end
            sessDirs{s} = sessDir;
        end

        mod = cell(numel(sessions), nCond);
        for s = 1:numel(sessions)
            for c = 1:nCond
                mod{s, c}{1}.name = '';
                mod{s, c}{1}.P = [];
                mod{s, c}{1}.h = 1;
            end
        end

        cov = cell(1, numel(sessions));
        cov_name = cell(1, numel(sessions));
        for s = 1:numel(sessions)
            mf = dir(fullfile(sessDirs{s}, 'motion*.txt'));
            if isempty(mf)
                error('no motion file in %s', sessDirs{s});
            elseif numel(mf) > 1
                warning('%s: %d motion files, using %s', subjID, numel(mf), mf(1).name);
            end
            mov = load(fullfile(sessDirs{s}, mf(1).name));
            if size(mov, 2) < 12
                error('motion file has %d columns, need 12', size(mov, 2));
            end
            cov{s} = mov(:, 1:12);
            cov_name{s} = {'trans_x', 'trans_y', 'trans_z', ...
                           'rot_x', 'rot_y', 'rot_z', ...
                           'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1', ...
                           'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1'};
        end

        glmDir = fullfile(glmRoot, subjID);
        if exist(glmDir, 'dir'), rmdir(glmDir, 's'); end
        mkdir(glmDir);
        cd(glmDir);

        SPM = makeSPMdesignmatrix_mov_reg(sessDirs, ons, ons_name, dur, mod, ...
                                          cov, cov_name, glmDir, TR);

        maskFile = fullfile(maskRoot, [subjID '_brain_mask.nii']);
        if ~exist(maskFile, 'file')
            error('brain mask missing: %s', maskFile);
        end
        Vm = spm_vol(maskFile);
        Vy = spm_vol(deblank(SPM.xY.P(1, :)));
        if ~isequal(Vm.dim, Vy.dim) || ~isequal(Vm.mat, Vy.mat)
            error('brain mask grid differs from the functional data');
        end
        SPM.xM.TH = -Inf(size(SPM.xM.TH));
        SPM.xM.VM = Vm;
        SPM.xM.I  = 0;
        SPM.xM.xs.Masking = 'explicit brain mask, no analysis threshold';

        spm_spm(SPM);

        shapeKey = strjoin(ons_name(:)', ' + ');
        shapes{end + 1} = shapeKey; %#ok<SAGROW>
        done = done + 1;
        Ymask = spm_read_vols(spm_vol(fullfile(glmDir, 'mask.nii')));
        nInMask = sum(Ymask(:) > 0);
        fprintf('%s  %d conditions, %d voxels modelled: %s\n', ...
                subjID, nCond, nInMask, shapeKey);

    catch ME
        warning('FAILED %s: %s', subjID, ME.message);
        failed{end + 1} = subjID; %#ok<SAGROW>
    end
end

fprintf('\n%s\n', repmat('=', 1, 72));
fprintf('estimated: %d/%d\n', done, nSub);
uniqueShapes = unique(shapes);
for i = 1:numel(uniqueShapes)
    fprintf('  %3d subjects: %s\n', sum(strcmp(shapes, uniqueShapes{i})), uniqueShapes{i});
end
if ~isempty(failed)
    fprintf('  FAILED: %s\n', strjoin(failed, ', '));
end
fprintf('%s\n', repmat('=', 1, 72));
