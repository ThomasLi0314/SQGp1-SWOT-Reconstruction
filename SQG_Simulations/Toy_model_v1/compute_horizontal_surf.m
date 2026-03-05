function [usurf, vsurf] = compute_horizontal_surf(Phi0, F1, G1, Phi1, kx, ky, Ro, z, dx, dz)
    Phi0_y = diff_spectral(Phi0, ky, 2);
    Phi0_x = diff_spectral(Phi0, kx, 1);
    Phi1_x = diff_spectral(Phi1, kx, 1);
    Phi1_y = diff_spectral(Phi1, ky, 2);

    [~, ~, F1_z] = gradient(F1, dx, dx, dz);
    [~, ~, G1_z] = gradient(G1, dx, dx, dz);

    u_3d = - Phi0_x - Ro * (Phi1_y + F1_z);
    v_3d = Phi0_y + Ro * (Phi1_x - G1_z);

    usurf = u_3d(:,:,end);
    vsurf = v_3d(:,:,end);
end