function w = compute_w(F1, G1, kx, ky, Ro)
    F1_x = diff_spectral(F1, kx, 1);
    G1_y = diff_spectral(G1, ky, 2);
    w = Ro * (F1_x + G1_y);
end