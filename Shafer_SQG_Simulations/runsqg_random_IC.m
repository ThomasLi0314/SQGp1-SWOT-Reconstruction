clear;
clc;
close all;


kmax = 255;  % Equivalent to 256^2 in grid space
K0 = 9.5;     % Initial wavenumber peak
delk = 3;    % Width of initial peak
E0 = 1;      % Initial energy
L = 2*pi;    % The domain scale is set to 2 pi, which means the
             % wavenumebers are integers; kx = 1 is a wave with one
             % wavelength in x fitting exactly into the domain

n = 2*(kmax+1);   % Grid-space resolution

% Initialize a grid of wavenumbers.  In this older version of my
% code, I use the wrapper functions grid2spec and spec2grid which
% first call fftshift before calling fft or ifft.  This orients the
% fields on a Cartesian wavenumber grid. Moreover, I compute things
% only in the upper-half-plane, since the lower-half-plane is given
% by conjugate symmetry:  psi(kx,ky) = -conj(psi(-kx,-ky))

% Note that in SQG, the spectral buoyancy bk and spectral
% streamfunction psik are related as bk = K_.*psik, where K_ =
% sqrt(kx.^2+ky.^2).  To invert from buoyancy, psik = bk./K_   
% This will give a NaN at (kx,ky) = (0,0).   In a doubly periodic
% model, the mean flow (0,0 wavenumber) doesn't change, so (0,0) is
% irrelevant to the model. I therefore set K_(kx=0,ky=0) = 1 so
% that I can invert without getting a NaN.

[kx_,ky_] = ndgrid(-kmax:kmax,0:kmax);
K_ = sqrt(kx_.^2+ky_.^2);     
K_(kmax+1,1) = 1;             % Array position of (0,0)

% Now initialize streamfunction in spectral space, with randomized
% phases distributed with Gaussian width delk about central
% wavenumber magnitude K0 (see expression f)

rng(10);                                        % Random generator seed for reproducibility 
f = exp(-(K_-K0).^2/delk^2);
psik = sqrt(f)./K_.*exp(2*pi*1i*rand(size(f))); % Initial psi field                       
e = real(sum(sum(K_.*psik.*conj(psik))));       % Energy in initial psi 
psik = E0*psik/sqrt(e);                         % Renormalize energy to E0

% Get inital buoyancy in grid space

bin = spec2grid(K_.*psik);

% Set run parameters

numsteps = 50000;   % Total number of timesteps
savestep = 1000;    % Save model output every 1000 timesteps
dttune = .05;       % Nondim tuning factor for timestepping 
hv = 8;             % Hyperviscosity:   nu*\grad^{hv}
nutune = .01;       % Nondim tuning factor for nu 

% Run the simulation.  Outputs are:
% bout(:,:,:)      Gridspace buoyancy with size (n,n,frame), where
%                  frame is approximately numsteps/savestep 
% time(:)          Time at which output is saved (frame)
% ke(:)            Timeseries of energy (frame)
% b2spec(:,:)      Spectra of bk*conj(bk) at each savestep (kmax,frame)

[bout,time,ke,b2spec] = sqg(bin,numsteps,savestep,dttune,nutune);

%% --- Export and Post-Processing ---
% 1. Create directory structure
sim_type = 'random_IC';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
folder_name = sprintf('%s_Shafer_Simulation_%s', timestamp, sim_type);
out_dir = fullfile('d:\Documents\College\Research\Oceangrophy\Shafer_Project\SQG_Simulations', 'Shafer Simulation output', folder_name);

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% 2. Save raw output data
data_file = fullfile(out_dir, sprintf('%s_Shafer_Simulation_%s.mat', timestamp, sim_type));
save(data_file, 'bout', 'time', 'ke', 'b2spec', 'kmax', 'K0', 'delk', 'E0', 'L', 'n', 'numsteps', 'savestep');
fprintf('Data saved to %s\n', data_file);

% Plot spectra, but skip initial condition 
figure(1)
clf
loglog(real(b2spec(:,2:end)))
grid
axis([1 100 1e-7 1])
title('Buoyancy Variance Spectrum')

% 3. Make a movie of buoyancy and save to file
figure(2)
clf
vidObj = VideoWriter(fullfile(out_dir, sprintf('%s_Shafer_Simulation_%s_animation.mp4', timestamp, sim_type)), 'MPEG-4');
vidObj.FrameRate = 10; % You can adjust this framerate
open(vidObj);

for j=1:size(bout,3)
    pcolor(bout(:,:,j)), shading interp, axis image
    colormap(jet);
    colorbar;
    title(sprintf('Buoyancy t = %f', time(j)));
    drawnow;
    % Write each frame to the video
    currFrame = getframe(gcf);
    writeVideo(vidObj, currFrame);
end
close(vidObj);
fprintf('Animation saved.\n');


% 4. Calculate Vorticity and Strain for the last frame
b_final = bout(:,:,end);
bk_final = grid2spec(b_final);

% In SQG, psi_k = b_k / K
% K_ was defined earlier, ensure we use the same array.
[kx_,ky_] = ndgrid(-kmax:kmax,0:kmax);
K_ = sqrt(kx_.^2+ky_.^2);     
K_(kmax+1,1) = 1; % Prevent NaN at (0,0)

psik = bk_final ./ K_;
psik(kmax+1,1) = 0; % Ensure mean streamfunction is zero

% Multiply by i*kx or i*ky to get derivatives in spectral space
% Vorticity zeta = \nabla^2 psi = - (kx^2 + ky^2) psi_k
% Since we have the SQG relation b_k = K_ * psi_k
% zeta_k = - K_^2 * psi_k = - K_ * b_k
zeta_k = - (K_.^2) .* psik; 
zeta = spec2grid(zeta_k);

% Strain components
% normal strain Sn = u_x - v_y = -psi_xy - psi_yx = -2 * psi_xy
Sn_k = -2 * (-kx_ .* ky_) .* psik; % (ikx)*(iky)*psik = -kx*ky*psik
Sn = spec2grid(Sn_k);

% shear strain Ss = v_x + u_y = psi_xx - psi_yy
Ss_k = -(kx_.^2 - ky_.^2) .* psik;
Ss = spec2grid(Ss_k);

% Total strain magnitude S = sqrt(Sn^2 + Ss^2)
S = sqrt(Sn.^2 + Ss.^2);

% 5. Plot Vorticity-Strain JPDF
figure(3)
clf

% Flatten the arrays for histogram processing
zeta_flat = zeta(:);
S_flat = S(:);

% Create a 2D histogram (JPDF)
% Use histcounts2 or similar. If histcounts2 is not available (older MATLAB),
% we can use a scatter plot with density coloring or a simple hist3.
nbins = 100;
[N, Xedges, Yedges] = histcounts2(zeta_flat, S_flat, nbins, 'Normalization', 'pdf');
% Centers for plotting
Xcenters = Xedges(1:end-1) + diff(Xedges)/2;
Ycenters = Yedges(1:end-1) + diff(Yedges)/2;

% Plot JPDF as a contour or pcolor
% Adding logarithmic scaling for probability density to match the reference plot
N(N==0) = NaN; % Set 0s to NaN so they don't plot in log scale
pcolor(Xcenters, Ycenters, log10(N)'); 
shading flat;
colormap(hot);
c = colorbar;
% Custom tick labels for log scale
ticks = get(c, 'Ticks');
set(c, 'TickLabels', arrayfun(@(x) sprintf('10^{%d}', x), ticks, 'UniformOutput', false));
c.Label.String = 'Probability Density';
xlabel('Vorticity (\zeta)');
ylabel('Strain (S)');
title('Vorticity-Strain JPDF (Last Time Step)');
axis tight;

hold on;
% Get current axis limits to draw the lines
ax = gca;
xlims = ax.XLim;
ylims = ax.YLim;

% Draw the Strain = Vorticity and Strain = -Vorticity lines
% S = zeta (for zeta > 0)
plot([0, max(xlims)], [0, max(xlims)], 'k--', 'LineWidth', 1.5);
% S = -zeta (for zeta < 0)
plot([min(xlims), 0], [abs(min(xlims)), 0], 'k--', 'LineWidth', 1.5);
hold off;

% Save the JPDF figure
fig_name = fullfile(out_dir, sprintf('%s_Shafer_Simulation_%s_JPDF.png', timestamp, sim_type));
saveas(gcf, fig_name);
fprintf('JPDF plot saved to %s\n', fig_name);
