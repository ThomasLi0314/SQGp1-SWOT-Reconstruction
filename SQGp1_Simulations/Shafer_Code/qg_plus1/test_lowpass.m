load('data_0508.mat');
Kc = 30;

% Buggy version (as-is)
flp_bug = lowpass(zeta0, Kc);

% Correct version (inline)
fk = g2k(zeta0);
[nkx, nky] = size(fk);
kmax = nky - 1;
[kx_, ky_] = ndgrid([0:kmax -kmax-1:-1], 0:kmax);
K_ = sqrt(kx_.^2 + ky_.^2);
mask = (K_ < Kc);
flp_fix = k2g(mask .* fk);

fprintf('rms zeta0          = %g\n', sqrt(mean(zeta0(:).^2)));
fprintf('rms flp_bug        = %g  (should be <= rms zeta0)\n', sqrt(mean(flp_bug(:).^2)));
fprintf('rms flp_fix        = %g\n', sqrt(mean(flp_fix(:).^2)));
fprintf('corr(zeta0,flp_bug)= %g  (should be ~1 for a true LP filter)\n', ...
    corr(zeta0(:), flp_bug(:)));
fprintf('corr(zeta0,flp_fix)= %g\n', corr(zeta0(:), flp_fix(:)));

f = figure('visible','off','position',[100 100 1500 450]);
subplot(1,3,1); imagesc(zeta0);    axis image; colorbar; title('zeta0');
subplot(1,3,2); imagesc(flp_bug);  axis image; colorbar; title('lowpass.m (buggy)');
subplot(1,3,3); imagesc(flp_fix);  axis image; colorbar; title('fixed: mask .* fk');
saveas(f, 'test_lowpass.png');
disp('saved test_lowpass.png');
exit;
