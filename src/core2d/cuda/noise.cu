// 
// Copyright(c) 1993-1996 Tony Kirke
/*
 * SPUC - Signal processing using C++ - A DSP library
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/
// Gaussian noise routine
// Can create (double) gaussian white noise
//            complex<double> gaussian white noise
//        or  (double) uniform noise
#include <math.h> 
#include <cstdlib>
#include <ctime>
#include "noise.h"
double noise::gauss()
	{
	double fac, r, v1, v2;

		if (s == 0) {
			do {
				v1 = (2.0 * uni()) - 1.0;
				v2 = (2.0 * uni()) - 1.0;
				r = (v1*v1) + (v2*v2);
			} while (r >= 1.0);
			fac = sqrt(-2.0 * log(r) / r);
			x = v1 * fac;
			s = 1;
			return (v2*fac);
		} else {
			s = 0;
			return (x);
		}
	}       
double noise::uni(void)
//--------------------------------------------------------------------
//       Returns uniform deviate between 0.0 and 1.0.
//       Used to generate PN data
//---------------------------------------------------------------------

	{
	double rm,r1;
	rm  = 1./M;
	idum = (long)fmod((float)IA*idum+IC,(float)M);
	r1 = idum*rm;
	return(r1);
}

void noise::whiteGauss(int numSamples)
{
	/* Generate a new random seed from system time - do this once in your constructor */
	srand(time(0));

	/* Setup constants */
	const static int q = 15;
	const static float c1 = (1 << q) - 1;
	const static float c2 = ((int)(c1 / 3)) + 1;
	const static float c3 = 1.f / c1;

	/* random number in range 0 - 1 not including 1 */
	float random = 0.f;

	/* the white noise */
	float noise = 0.f;

	for (int i = 0; i < numSamples; i++)
	{
		random = ((float)rand() / ((float)RAND_MAX + 1.f));
		noise = (2.f * ((random * c2) + (random * c2) + (random * c2)) - 3.f * (c2 - 1.f)) * c3;
	}
}