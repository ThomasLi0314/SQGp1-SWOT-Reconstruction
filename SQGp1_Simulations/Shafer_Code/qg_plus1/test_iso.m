load('data_0508.mat');

s0     = isospectrum(zeta0);
s1_Ro2 = isospectrum(Ro*zeta1);     % Ro^2 power
s1_raw = isospectrum(zeta1);        % no Ro
s1_Ro  = Ro * isospectrum(zeta1);   % linear Ro
s1     = s1_Ro2;                    % default for the rest of the script
fprintf('peak s1_raw = %g  s1_Ro = %g  s1_Ro2 = %g\n', max(s1_raw), max(s1_Ro), max(s1_Ro2));

fprintf('size(zeta0) = [%d %d]\n', size(zeta0));
fprintf('Ro = %g\n', Ro);
fprintf('size(s0) = [%d %d]\n', size(s0));
fprintf('size(s1) = [%d %d]\n', size(s1));
fprintf('s0(1:5) = '); fprintf('%g ', s0(1:5)); fprintf('\n');
fprintf('s1(1:5) = '); fprintf('%g ', s1(1:5)); fprintf('\n');
fprintf('any(isnan(s0)) = %d\n', any(isnan(s0)));
fprintf('min(s0)=%g  max(s0)=%g\n', min(s0), max(s0));
fprintf('min(s1)=%g  max(s1)=%g\n', min(s1), max(s1));

K = 1:length(s0);
f = figure('visible','off');
loglog(K, s0, K, s1_raw, K, s1_Ro, K, s1_Ro2);
grid on;
ylim([1e-6 1e2]);
legend('\zeta_0','iso(\zeta_1)','Ro\cdot iso(\zeta_1)','iso(Ro\cdot\zeta_1)');
saveas(f, 'test_iso_out3.png');
disp('saved test_iso_out.png');
exit;
