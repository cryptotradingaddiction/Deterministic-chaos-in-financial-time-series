set terminal wxt size 900,600 enhanced font 'Arial,12'
set title "Odhad největšího Ljapunovova exponentu (Rosenstein) – ADAUSD"
set xlabel "Časový krok i"
set ylabel "<ln(divergence)>"
set grid
set key top left

# Pouze pro dva sloupce (i, S)
plot "rosenstein_ada" using 1:2 with linespoints lw 1.5 pt 7 ps 0.7 title "S(i)"

pause -1