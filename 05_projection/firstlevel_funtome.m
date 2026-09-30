% First-level GLM (SPM12) on Functionnectome-projected data, one priors set.
%
% IN   setLabel                          proj, asso, wb or comm, set before calling
%      subjectFile                       optional, default subjects_stopsignal_all.txt, one participant ID per line, in this folder
%      stopsignal/functionnectome/<set>/voxelwise_analysis/<sub>/functionnectome.nii.gz   made by run_projection.sh
%      stopsignal/staged/<sub>/sess1/sess1.mat          made by extract_onsets.py (01_greymatter_glm)
%      stopsignal/staged/<sub>/sess1/motion_sess1.txt   made by extract_motion_parameters.py (01_greymatter_glm)
%      priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI
%      stopsignal/glm_all/<sub>/SPM.mat                 made by firstlevel_greymatter.m (01_greymatter_glm), design comparison only
%      makeSPMdesignmatrix_mov_reg.m                    in 01_greymatter_glm
%
% OUT  stopsignal/functionnectome/<set>/glm/<sub>/{SPM.mat, beta_*.nii, ResMS.nii, mask.nii, mask_input.nii}
%      stopsignal/functionnectome/<set>/glm/subjects_estimated.txt

% ============================== CONFIG ==============================
spmPath       = getenv('SPM_PATH');
projectRoot   = getenv('PROJECT_ROOT');
scriptDir     = fileparts(mfilename('fullpath'));
if ~exist('setLabel', 'var')
    error('set setLabel (proj, asso, wb or comm) before calling');
end
if ~exist('subjectFile', 'var')
    subjectFile = fullfile(scriptDir, 'subjects_stopsignal_all.txt');
end
funtomeRoot   = fullfile(projectRoot, 'stopsignal', 'functionnectome', setLabel);
stagedRoot    = fullfile(projectRoot, 'stopsignal', 'staged');
templateGz    = fullfile(projectRoot, 'priors', 'priors_template.nii.gz');
classicalRoot = fullfile(projectRoot, 'stopsignal', 'glm_all');
glmRoot       = fullfile(funtomeRoot, 'glm');
TR            = 2.0;
deleteUnzipped = true;
% ====================================================================

addpath(spmPath);
addpath(scriptDir);
if ~exist('spm.m', 'file')
    error('SPM12 not found at %s', spmPath);
end
spm('Defaults', 'fMRI');
spm_jobman('initcfg');
spm_get_defaults('cmdline', true);

subj_ids = read_list(subjectFile);
nSub = numel(subj_ids);
if nSub == 0
    error('no subjects read from %s', subjectFile);
end
fprintf('set:       %s\n', setLabel);
fprintf('subjects:  %d, read from %s\n', nSub, subjectFile);
fprintf('glm out:   %s\n', glmRoot);
fprintf('TR:        %.2f s, delete unzipped input: %d\n\n', TR, deleteUnzipped);

if ~exist(glmRoot, 'dir'), mkdir(glmRoot); end
templateFile = fullfile(glmRoot, 'priors_template.nii');
if ~exist(templateFile, 'file')
    gunzip(templateGz, glmRoot);
end
Vt = spm_vol(templateFile);
template = spm_read_vols(Vt) > 0;
fprintf('template voxels: %d\n\n', nnz(template));

cov_names = {'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', ...
             'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1', ...
             'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1'};

done = {};
failed = {};

for subj = 1:nSub
    subjID = subj_ids{subj};
    tSub = tic;
    fprintf('[%d/%d] %s  started %s\n', subj, nSub, subjID, datestr(now, 'HH:MM:SS'));
    try
        srcFile = fullfile(funtomeRoot, 'voxelwise_analysis', subjID, 'functionnectome.nii.gz');
        stagedDir = fullfile(stagedRoot, subjID, 'sess1');
        if ~exist(srcFile, 'file'), error('projected input missing: %s', srcFile); end

        L = load(fullfile(stagedDir, 'sess1.mat'), 'onsets', 'durations', 'names');
        nCond = numel(L.names);
        ons = L.onsets(:);
        dur = L.durations(:);
        ons_name = L.names(:);
        mod = cell(1, nCond);
        for c = 1:nCond
            mod{1, c}{1}.name = '';
            mod{1, c}{1}.P = [];
            mod{1, c}{1}.h = 1;
        end
        mov = load(fullfile(stagedDir, 'motion_sess1.txt'));
        if size(mov, 2) < 12
            error('motion file has %d columns, need 12', size(mov, 2));
        end

        glmDir = fullfile(glmRoot, subjID);
        cd(glmRoot);
        if exist(glmDir, 'dir'), rmdir(glmDir, 's'); end
        sessDir = fullfile(glmDir, 'sess1');
        mkdir(sessDir);
        t = tic;
        gunzip(srcFile, sessDir);
        funcFile = fullfile(sessDir, 'functionnectome.nii');
        Vy = spm_vol(funcFile);
        fprintf('  unzipped, %d volumes (%.0f s)\n', numel(Vy), toc(t));
        if numel(Vy) ~= size(mov, 1)
            error('%d volumes but %d motion rows', numel(Vy), size(mov, 1));
        end
        if ~isequal(Vy(1).dim, Vt.dim) || max(abs(Vy(1).mat(:) - Vt.mat(:))) > 1e-4
            error('projected data and priors template are on different grids');
        end

        t = tic;
        Y = spm_read_vols(Vy);
        varying = any(Y ~= Y(:, :, :, 1), 4);
        clear Y
        inMask = template & varying;
        Vm = Vt;
        Vm.fname = fullfile(glmDir, 'mask_input.nii');
        Vm.dt = [2 0];
        Vm.pinfo = [1; 0; 0];
        Vm.n = [1 1];
        spm_write_vol(Vm, double(inMask));
        fprintf('  mask: %d template voxels, %d constant excluded, %d kept (%.0f s)\n', ...
                nnz(template), nnz(template & ~varying), nnz(inMask), toc(t));

        cd(glmDir);
        SPM = makeSPMdesignmatrix_mov_reg({sessDir}, ons, ons_name, dur, mod, ...
                                          {mov(:, 1:12)}, {cov_names}, glmDir, TR);
        SPM.xM.TH = -Inf(size(SPM.xM.TH));
        SPM.xM.VM = spm_vol(Vm.fname);
        SPM.xM.I  = 0;
        SPM.xM.xs.Masking = 'explicit: priors template minus constant voxels, no analysis threshold';

        t = tic;
        SPM = spm_spm(SPM);
        Ymask = spm_read_vols(spm_vol(fullfile(glmDir, 'mask.nii')));
        fprintf('  estimated, %d voxels modelled (%.0f s)\n', nnz(Ymask > 0), toc(t));

        classicalFile = fullfile(classicalRoot, subjID, 'SPM.mat');
        if exist(classicalFile, 'file')
            C = load(classicalFile, 'SPM');
            C = C.SPM;
            sameNames = isequal(C.xX.name, SPM.xX.name);
            if isequal(size(C.xX.X), size(SPM.xX.X))
                dX = max(abs(C.xX.X(:) - SPM.xX.X(:)));
            else
                dX = NaN;
            end
            if isfield(C.xX, 'W') && isequal(size(C.xX.W), size(SPM.xX.W))
                dW = norm(full(SPM.xX.W - C.xX.W), 'fro') / norm(full(C.xX.W), 'fro');
            else
                dW = NaN;
            end
            fprintf('  vs grey-matter GLM: names identical %d, columns %d vs %d, design max abs diff %.3g, whitening relative diff %.3g\n', ...
                    sameNames, size(SPM.xX.X, 2), size(C.xX.X, 2), dX, dW);
        else
            fprintf('  vs grey-matter GLM: no SPM.mat at %s\n', classicalFile);
        end

        if deleteUnzipped
            delete(funcFile);
        end
        done{end + 1} = subjID; %#ok<SAGROW>
        fprintf('  %s done, %d conditions, %.0f s total\n', subjID, nCond, toc(tSub));

    catch ME
        warning('FAILED %s: %s', subjID, ME.message);
        failed{end + 1} = subjID; %#ok<SAGROW>
    end
end

fid = fopen(fullfile(glmRoot, 'subjects_estimated.txt'), 'w');
fprintf(fid, '%s\n', done{:});
fclose(fid);

fprintf('\n%s\n', repmat('=', 1, 72));
fprintf('estimated: %d/%d\n', numel(done), nSub);
fprintf('list:      %s\n', fullfile(glmRoot, 'subjects_estimated.txt'));
if ~isempty(failed)
    fprintf('FAILED:    %s\n', strjoin(failed, ', '));
end
fprintf('%s\n', repmat('=', 1, 72));


function ids = read_list(fname)
fid = fopen(fname, 'r');
if fid < 0
    error('cannot open subject list: %s', fname);
end
ids = {};
while true
    ln = fgetl(fid);
    if ~ischar(ln), break; end
    ln = strtrim(ln);
    if isempty(ln) || ln(1) == '#', continue; end
    ids{end + 1} = ln; %#ok<AGROW>
end
fclose(fid);
end
