set terminal pngcairo size 1400,900
set grid
set key top right width 2 box
# Spolecne labely
set label 1 "ACF nula (delay): 2" at screen 0.78,0.92 font "Arial,12" tc rgb "blue"
set label 2 "Theiler W: " at screen 0.78,0.86 font "Arial,12" tc rgb "dark-green"
# --- GRAF 1: ACF ---
set output "acf_plot.png"
set title "Autocorrelation Function - ADAUSD"
set xlabel "Delay (time units)"
set ylabel "Autocorrelation"
set xrange [0:100]
set yrange [-0.2:1.1]
set arrow from 2, graph 0 to 2, graph 1 nohead lc rgb "red" dt 2
set arrow from , graph 0 to , graph 1 nohead lc rgb "dark-green" dt 2
plot "acf_out.dat" using 1:2 with lines lw 2 title "ACF", 0 with lines lc rgb "black" notitle
# --- GRAF 2: STP LINEARNI ---
unset arrow
unset label
set output "stp_plot_linear.png"
set title "Space-Time Separation Plot - ADAUSD (Linearni)"
set xlabel "Relative time distance"
set ylabel "Spatial distance"
set autoscale x
set autoscale y
unset logscale y
set key top left
plot "stp_out_all" every :::0::0  with lines lw 1.5 title "1%", echo      "stp_out_all" every :::9::9  with lines lw 1.5 title "10%", echo      "stp_out_all" every :::49::49 with lines lw 1.5 title "50%", echo      "stp_out_all" every :::89::89 with lines lw 1.5 title "90%", echo      "stp_out_all" every :::98::98 with lines lw 1.5 title "99%"
# --- GRAF 3: STP LOGARITMICKY ---
set output "stp_plot_log.png"
set title "Space-Time Separation Plot - ADAUSD (Logaritmicky)"
set logscale y
plot "stp_out_all" every :::0::0  with lines lw 1.5 title "1%", echo      "stp_out_all" every :::9::9  with lines lw 1.5 title "10%", echo      "stp_out_all" every :::49::49 with lines lw 1.5 title "50%", echo      "stp_out_all" every :::89::89 with lines lw 1.5 title "90%", echo      "stp_out_all" every :::98::98 with lines lw 1.5 title "99%"
