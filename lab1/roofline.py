import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

GIGA = 1000000000

def plot_roofline(peak_gflops, bw, ai, g1, g2):
	#find a.i. for which FLOPS = peak flops using bw
	x = peak_gflops / bw

	print(x)
	peak_flops = peak_gflops * GIGA
	max_x = max(ai, x) + x/3
	# plot the peak flops line
	X = [x, max_x]
	Y = [peak_flops, peak_flops]
	line1, = plt.plot(X, Y, color = "red", label = "CPU " + str(peak_gflops) + " GFLOPS")

	#plot the dram bw line
	X = [0, x]
	Y = [0, peak_flops]
	line2, = plt.plot(X, Y, color = "blue", label = "DRAM BW " + str(bw) + " GB/s")


	#plot the arithmetic intensity vertical lines
	X = [ai, ai]
	Y = [0, 1.1 * peak_flops]
	line3, = plt.plot(X, Y, color = "black", linestyle = "dashed", label = "Benchmark A.I. " + str(ai) + " GFLOP/byte")

	X = [x, x]
	line4 = plt.plot(X, Y, color = "green", linestyle = "dashed")

	#plot the gflops measurments
	X = ai
	Y = g1 * GIGA
	line5, = plt.plot(X, Y, color = "orange", marker = 'x', markeredgewidth = 2.0, linestyle = '', label = "Benchmark 1 " + str(g1) + " gflops")
	
	X = ai
	Y = g2 * GIGA
	line6, = plt.plot(X, Y, color = "purple", marker = 'x', markeredgewidth = 2.0, linestyle = '', label = "Benchmark 2 " + str(g2) + " gflops")

	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel("Arithmetic Intensity [FLOP/byte]")
	plt.ylabel("FLOPS")
	plt.legend(handles = [line1, line2, line3, line5, line6], loc = "upper left", fontsize = "small", numpoints = 1)
	plt.savefig('Q3.png')
	plt.close()



def main():
	parser = argparse.ArgumentParser(description = "Plot roofline model.")
	parser.add_argument("peak_gflops", type=float, help = "CPU peak GFLOPS")
	parser.add_argument("bw", type=float, help = "DRAM bandwidth in GB/s")
	parser.add_argument("ai", type=float, help = "Arithmetic Intensity")
	parser.add_argument("g1", type=float, help = "GFLOP/s measurment 1")
	parser.add_argument("g2", type=float, help = "GFLOP/s measurment 2")
	args = parser.parse_args()

	plot_roofline(args.peak_gflops, args.bw, args.ai, args.g1, args.g2)


if __name__ == "__main__": 
	main()	
