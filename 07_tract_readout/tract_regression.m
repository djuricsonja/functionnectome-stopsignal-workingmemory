function tract_regression(designFile, colldiagDir)
% Spatial regression of each set's group-mean contrast map on the tract predictors, with collinearity
% diagnostics and each tract's adjusted R2 after residualising it on the others.
%
% IN   designFile                       stopsignal/functionnectome/tract_regression/design_<setA>_<setB>_<mask>_mean.mat
%                                       made by build_regression_design.py
%      colldiagDir                      optional, folder holding colldiag.m (github.com/brian-lau/colldiag, commit f5a0c2b)
% OUT  regression_<setA>_<setB>_<mask>_mean_{fit,betas,collinearity}.csv, beside the design file

% ============================== CONFIG ==============================
COLLDIAG_DIR = getenv('COLLDIAG_PATH');
CI_LIMIT     = 30;
VDP_LIMIT    = 0.5;
REPORT_R2    = 0.01;
% ====================================================================

if nargin < 1
    error('give the design file');
end
if nargin < 2, colldiagDir = COLLDIAG_DIR; end
if ~exist(fullfile(colldiagDir, 'colldiag.m'), 'file')
    error('colldiag.m not found in %s', colldiagDir);
end
addpath(colldiagDir);

[folder, stem] = fileparts(designFile);
stem = regexprep(stem, '^design_', 'regression_');
outFit = fullfile(folder, [stem '_fit.csv']);
outBeta = fullfile(folder, [stem '_betas.csv']);
outColl = fullfile(folder, [stem '_collinearity.csv']);
for f = {outFit, outBeta, outColl}
    if exist(f{1}, 'file'), error('output exists, not overwriting: %s', f{1}); end
end

D = load(designFile);
X = D.X;
Y = D.Y;
tract = as_cellstr(D.tract);
source = as_cellstr(D.source);
sets = as_cellstr(D.sets);
contrasts = as_cellstr(D.contrasts);
[n, p] = size(X);
if numel(tract) ~= p, error('%d tract names for %d columns', numel(tract), p); end
fprintf('design %s\nvoxels %d   predictors %d\n', designFile, n, p);

Xc = X - mean(X, 1);
fprintf('rank of centred X: %d of %d\n', rank(Xc), p);
info = colldiag(Xc, tract, VDP_LIMIT, false, true);
ci = info.condind(:);
vdp = info.vdp;                                  % dimensions x tracts
fprintf('\ncondition indices: max %.2f   above %g: %d\n', max(ci), CI_LIMIT, nnz(ci > CI_LIMIT));
nFlag = 0;
[~, order] = sort(ci, 'descend');
for k = order(:)'
    hit = find(vdp(k, :) >= VDP_LIMIT);
    if ci(k) > CI_LIMIT && numel(hit) >= 2
        nFlag = nFlag + 1;
        fprintf('   FLAGGED dimension %d  CI %.2f\n', k, ci(k));
        for j = hit, fprintf('      %-45s %.3f\n', tract{j}, vdp(k, j)); end
    end
end
fprintf('flagged dimensions: %d\n', nFlag);
dimCol = repmat((1:p)', 1, p); ciCol = repmat(ci, 1, p); trCol = repmat(tract(:)', p, 1);
Tc = table(dimCol(:), ciCol(:), trCol(:), vdp(:), ...
           'VariableNames', {'dimension', 'condition_index', 'tract', 'vdp'});
writetable(Tc, outColl);

R = zeros(n, p);
for j = 1:p
    others = setdiff(1:p, j);
    m = fitlm(X(:, others), X(:, j));
    R(:, j) = m.Residuals.Raw;
end

fitRows = {}; betaRows = {};
for a = 1:numel(sets)
    for b = 1:numel(contrasts)
        y = Y(:, a, b);
        mdl = fitlm(X, y);
        est = mdl.Coefficients;
        r2res = zeros(p, 1);
        for j = 1:p
            m = fitlm(R(:, j), y);
            r2res(j) = m.Rsquared.Adjusted;
        end
        fitRows(end + 1, :) = {sets{a}, contrasts{b}, n, mdl.Rsquared.Ordinary, mdl.Rsquared.Adjusted, ...
                               mdl.RMSE, est.Estimate(1)}; %#ok<AGROW>
        for j = 1:p
            betaRows(end + 1, :) = {sets{a}, contrasts{b}, tract{j}, source{j}, est.Estimate(j + 1), ...
                                    est.SE(j + 1), est.tStat(j + 1), est.pValue(j + 1), r2res(j), ...
                                    r2res(j) > REPORT_R2, info.vif(j)}; %#ok<AGROW>
        end
        fprintf('\n%-5s %-24s R2 %.4f  adjusted R2 %.4f  RMSE %.3f  intercept %+.3f  tracts above %g: %d\n', ...
                sets{a}, contrasts{b}, mdl.Rsquared.Ordinary, mdl.Rsquared.Adjusted, mdl.RMSE, ...
                est.Estimate(1), REPORT_R2, nnz(r2res > REPORT_R2));
    end
end
writetable(cell2table(fitRows, 'VariableNames', {'set', 'contrast', 'voxels', 'r2', 'r2_adjusted', ...
           'rmse', 'intercept'}), outFit);
writetable(cell2table(betaRows, 'VariableNames', {'set', 'contrast', 'tract', 'source', 'beta', 'se', ...
           't', 'p', 'r2adj_residualised', 'above_report_cut', 'vif'}), outBeta);
fprintf('\nwrote %s\nwrote %s\nwrote %s\n', outFit, outBeta, outColl);
end


function c = as_cellstr(v)
if ischar(v), c = cellstr(v); else, c = cellfun(@strtrim, cellstr(v), 'UniformOutput', false); end
c = c(:)';
end
