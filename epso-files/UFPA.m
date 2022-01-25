function L = UFPA(x, ht, hr, f, d)
lamb = 3e8/f;
fM = f*1e-6;
L = x(1)*log10(d) + x(2)*log10(fM) + x(3) - x(4).*((ht + hr)*lamb)./(0.1*62);
end

