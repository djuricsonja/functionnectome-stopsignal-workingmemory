% Helper: builds one participant's SPM12 first-level design.
%
% IN   sessDirs{i}   folder holding exactly one 4D .nii
% OUT  <glmDir>/SPM.mat

function SPM = makeSPMdesignmatrix_mov_reg(sessDirs, ons, ons_name, ons_duration, ons_modulate, cov, cov_name, glmDir, TR)

NumSessions = length(sessDirs);
NumVar = length(ons_name);

SPM.xY.RT = TR;
SPM.xBF.name = 'hrf';
SPM.xBF.length = 32;
SPM.xBF.order = 1;
SPM.xBF.T = 16;
SPM.xBF.T0 = 1;
SPM.xBF.UNITS = 'secs';
SPM.xBF.Volterra = 1;

for i = 1:NumSessions
    counter = 1;
    for j = 1:NumVar
        if ~isempty(ons{j,i})
            SPM.Sess(i).U(counter).name = ons_name(j);
            SPM.Sess(i).U(counter).ons = ons{j,i};
            SPM.Sess(i).U(counter).dur = ons_duration{j,i};
            for k = 1:length(ons_modulate{i,j})
                SPM.Sess(i).U(counter).P(k).name = ons_modulate{i,j}{k}.name;
                SPM.Sess(i).U(counter).P(k).P = ons_modulate{i,j}{k}.P;
                SPM.Sess(i).U(counter).P(k).h = ons_modulate{i,j}{k}.h;
            end
            counter = counter + 1;
        end
    end
end

scanList = [];
SPM.nscan = zeros(1, NumSessions);

for i = 1:NumSessions
    sessionDir = sessDirs{i};

    gz_files = dir(fullfile(sessionDir, '*.nii.gz'));
    for k = 1:length(gz_files)
        nii_name = gz_files(k).name(1:end-3);
        if ~exist(fullfile(sessionDir, nii_name), 'file')
            gunzip(fullfile(sessionDir, gz_files(k).name));
        end
    end

    nii_files = dir(fullfile(sessionDir, '*.nii'));
    nii_files = nii_files(~startsWith({nii_files.name}, '._'));

    if isempty(nii_files)
        error('No usable .nii files found in %s after unzipping.', sessionDir);
    elseif length(nii_files) > 1
        error('Multiple .nii files found in %s: %s', sessionDir, strjoin({nii_files.name}, ', '));
    end

    fpath = fullfile(sessionDir, nii_files(1).name);
    V = spm_vol(fpath);

    sessionScans = strings(numel(V), 1);
    for v = 1:numel(V)
        sessionScans(v) = sprintf('%s,%d', fpath, v);
    end
    scanList = [scanList; sessionScans];
    SPM.nscan(i) = numel(V);
end

SPM.xY.P = char(scanList);

for i = 1:NumSessions
    SPM.Sess(i).C.C = cov{i};
    SPM.Sess(i).C.name = cov_name{i};
end

SPM.xGX.iGXcalc = 'None';
SPM.xX.K.HParam = 200;
SPM.xVi.form = 'AR(1) + w';

disp('Configuring the design matrix...');
SPM.swd = glmDir;
SPM = spm_fmri_spm_ui(SPM);
disp('Finished making the design matrix.');
