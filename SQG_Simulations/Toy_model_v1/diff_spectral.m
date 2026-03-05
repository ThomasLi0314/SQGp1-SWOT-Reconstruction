function out = diff_spectral(in, k_vec, ~)
    in_hat = fft2(in);
    multiplier = 1i * k_vec;
    if ndims(in) == 3
        multiplier = repmat(multiplier, [1, 1, size(in,3)]);
    end
    out_hat = in_hat .* multiplier;
    out = real(ifft2(out_hat));
end