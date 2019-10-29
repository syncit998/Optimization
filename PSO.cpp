#include <iostream>
#include <iomanip>
#include <math.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>

#include "MersenneTwister.h"
#include "CStopWatch.h"

using namespace std;
#define PI_F 3.141592654f 

typedef vector<int> 		iArray1D;
typedef vector<float> ldArray1D;
typedef vector<ldArray1D> 	ldArray2D;
typedef vector<ldArray2D> 	ldArray3D;

double sigMoid(double v){

	return 1/(1+exp(-v));
}

float F1(ldArray2D& R, int Nd, int p) {
	float Z=0,  Xi;

	for(int i=0; i<Nd; i++){
		Xi = R[p][i];
		Z += Xi*Xi;
	}

	return -Z;

}

float F2(ldArray2D& R, int Nd, int p) { // Sphere
	float Z=0,  Xi;

	for(int i=0; i<Nd; i++){
		Xi = R[p][i];
		Z += (pow(Xi,2) - 10 * cos(2*PI_F*Xi) + 10);
	}
	return -Z;
}
float F3(ldArray2D& R, int Nd, int p) { // Sphere
	float Z, Sum, Prod, Xi;

    Z = 0; Sum = 0; Prod = 1;
    
	for(int i=0; i<Nd; i++){
		Xi = R[p][i];
		Sum  += Xi*Xi;
		Prod *= cos(Xi/sqrt((double)i)+1)/4000.0f; 
		
		if(isnan(Prod)) Prod = 1;
    }
	
	Z = Sum - Prod;
	
	return -Z;
}
float F4(ldArray2D& R, int Nd, int p) {
	float Z=0, Xi, XiPlus1;
 
	for(int i=0; i<Nd-1; i++){
		Xi = R[p][i];
		XiPlus1 = R[p][i+1];
		
		Z = Z + (100*(XiPlus1-Xi*Xi)*(XiPlus1-Xi*Xi) + (Xi-1)*(Xi-1));
	}
	return -Z;
}

void SPSO2007(int Np, int Nd, int Nt, float xMin, float xMax, float vMin, float vMax,float (*objFunc)(ldArray2D& ,int, int), int& numEvals, string functionName){
    vector < vector< float> > R(Np, vector<float>(Nd, 0));
	vector < vector< float> > V(Np, vector<float>(Nd, 0));
	vector<float> M(Np,0);

	ldArray2D pBestPosition(Np, vector<float>(Nd,-INFINITY));
	ldArray1D pBestValue(Np,-INFINITY);
    
    int left, right;
    ldArray2D gBestPosition(Np, vector<float>(Nd, -INFINITY));
    ldArray1D gBestValue(Np,-INFINITY);
    float bestFitness = -INFINITY;
    
	int lastStep = Nt, bestTimeStep;

	MTRand mt;

	float  C1  = 2.05, C2 = 2.05;
	float  phi = C1 + C2;
	float  chi = 0.95;//2.0/fabs(2.0 - phi - sqrt(phi*phi - 4*phi));
	float  R1, R2;

	CStopWatch timer, timer1;
	float positionTime = 0, fitnessTime = 0, velocityTime = 0, totalTime = 0;;

	numEvals = 0;
    timer1.startTimer();
    
	// Init Population
	for(int p=0; p<Np; p++){
		for(int i=0; i<Nd; i++){
		    V[p][i] = vMin + mt.randDblExc(vMax-vMin);
			R[p][i] = xMin + mt.randDblExc(xMax-xMin);

			if(mt.rand() < 0.5){
				R[p][i] = -R[p][i];
			    V[p][i] = -V[p][i];    
			}
		}
	}

	// Evaluate Fitness
	for(int p=0; p<Np; p++){
		 M[p] = objFunc(R, Nd, p);
		 numEvals++;
	}

	for(int j=1; j<Nt; j++){

		//Update Positions
		timer.startTimer();
		for(int p=0; p<Np; p++){
			for(int i=0; i<Nd; i++){
				R[p][i] = R[p][i] + V[p][i];

				if(R[p][i] > xMax) R[p][i] = xMin + mt.randDblExc(xMax-xMin);
                if(R[p][i] < xMin) R[p][i] = xMin + mt.randDblExc(xMax-xMin);
			}
		}
		timer.stopTimer();
		positionTime += timer.getElapsedTime();

		// Evaluate Fitness
		timer.startTimer();
		for(int p=0; p<Np; p++){
			M[p] = objFunc(R, Nd, p);
			numEvals++;
		}
		
		for(int p=0; p<Np; p++){
		    left = (p-1);
		    if(p == 0) left = Np-1;
		    right = (p+1) % Np;
		    
		    // Global
		    if(M[left] > gBestValue[p]){
		        gBestValue[p] = M[left];
		        for(int i=0; i<Nd; i++){
				    gBestPosition[p][i] = R[p][i];
				}
		    }
		    if(M[p] > gBestValue[p]){
		        gBestValue[p] = M[p];
		        for(int i=0; i<Nd; i++){
				    gBestPosition[p][i] = R[p][i];
				}
		    }
		    if(M[right] > gBestValue[p]){
		        gBestValue[p] = M[right];
		        for(int i=0; i<Nd; i++){
				    gBestPosition[p][i] = R[p][i];
				}
		    }
		    
		    if(gBestValue[p] > bestFitness){
		        bestFitness = gBestValue[p];
		        bestTimeStep = j;
	        }
	        
	        //Personal Best
	        if(M[p] > pBestValue[p]){
	            pBestValue[p] = M[p];
		        for(int i=0; i<Nd; i++){
				    pBestPosition[p][i] = R[p][i];
				}
	        }
	        
		}
		timer.stopTimer();
		fitnessTime += timer.getElapsedTime();

        if(bestFitness >= -0.0001){
            lastStep = j;
    	    break;
	    }

		// Update Velocities
		timer.startTimer();
		
		for(int p=0; p<Np; p++){
			for(int i=0; i<Nd; i++){
    			R1 = mt.rand(); R2 = mt.rand();
    			
               // V[p][i] = chi * mt.rand() *  V[p][i] + C1*R1*(pBestPosition[p][i] - R[p][i]) + C2*R2*(gBestPosition[p][i] - R[p][i]);       
                V[p][i] = chi * (V[p][i] + C1*R1*(pBestPosition[p][i] - R[p][i]) + C2*R2*(gBestPosition[p][i] - R[p][i]));                     
                if(V[p][i] > vMax) V[p][i] = vMin + mt.randDblExc(vMax-vMin);
                if(V[p][i] < vMin) V[p][i] = vMin + mt.randDblExc(vMax-vMin);
			}
		}
		timer.stopTimer();
		velocityTime += timer.getElapsedTime();
	} // End Time Steps

    timer1.stopTimer();
    totalTime += timer1.getElapsedTime();

    R.clear(); V.clear(); M.clear();
    pBestPosition.clear(); pBestValue.clear(); gBestPosition.clear();
        
	cout    << functionName << ","
            << bestFitness  << "," 
            << Np << ","
            << Nd << ","
            << lastStep << ","
            << numEvals << ","
            << positionTime << ","
            << fitnessTime << ","
            << velocityTime << ","
            << totalTime << endl;
}
void PSO(int Np, int Nd, int Nt, float xMin, float xMax, float vMin, float vMax,float (*objFunc)(ldArray2D& ,int, int), int& numEvals, string functionName){

	vector < vector< float> > R(Np, vector<float>(Nd, 0));
	vector < vector< float> > V(Np, vector<float>(Nd, 0));
	vector<float> M(Np,0);

	ldArray2D pBestPosition(Np, vector<float>(Nd,-INFINITY));
	ldArray1D pBestValue(Np,-INFINITY);
	
	ldArray1D gBestPosition(Nd, -INFINITY);
    float gBestValue = -INFINITY;
    
	int lastStep = Nt, bestTimeStep;

	MTRand mt;
	

	float  C1  = 2.05, C2 = 2.05;
	float  phi = C1 + C2;
	float w, wMax = 0.9, wMin = 0.4;
	float  R1, R2;

	CStopWatch timer, timer1;
	float positionTime = 0, fitnessTime = 0, velocityTime = 0, totalTime = 0;;

	numEvals = 0;
    timer1.startTimer();
    
	// Init Population
	for(int p=0; p<Np; p++){
		for(int i=0; i<Nd; i++){
			R[p][i] = xMin + mt.randDblExc(xMax-xMin);
			V[p][i] = vMin + mt.randDblExc(vMax-vMin);

			if(mt.rand() < 0.5){
				R[p][i] = -R[p][i];
				V[p][i] = -V[p][i];
			}
		}
	}

	// Evaluate Fitness
	for(int p=0; p<Np; p++){
		 M[p] = objFunc(R, Nd, p);
		 numEvals++;
	}

	for(int j=1; j<Nt; j++){

		//Update Positions
		timer.startTimer();
		for(int p=0; p<Np; p++){
			for(int i=0; i<Nd; i++){
				R[p][i] = R[p][i] + V[p][i];

				if(R[p][i] > xMax) R[p][i] = xMin + mt.randDblExc(xMax-xMin);
                if(R[p][i] < xMin) R[p][i] = xMin + mt.randDblExc(xMax-xMin);
			}
		}
		timer.stopTimer();
		positionTime += timer.getElapsedTime();

		// Evaluate Fitness
		timer.startTimer();
		for(int p=0; p<Np; p++){
			M[p] = objFunc(R, Nd, p);
			numEvals++;
		}
		
		for(int p=0; p<Np; p++){
		    if(M[p] > gBestValue){
				gBestValue = M[p];
				for(int i=0; i<Nd; i++){
				    gBestPosition[i] = R[p][i];
				}
				bestTimeStep = j;
			}
			
			// Local
		    if(M[p] > pBestValue[p]){
			    pBestValue[p] = M[p];
                for(int i=0; i<Nd; i++){
				    pBestPosition[p][i] = R[p][i];
				}
			}
			
		}
		timer.stopTimer();
		fitnessTime += timer.getElapsedTime();

        if(gBestValue >= -0.0001){
            lastStep = j;
    	    break;
	    }

		// Update Velocities
		timer.startTimer();
		w = wMax - ((wMax-wMin)/Nt) * j;
		for(int p=0; p<Np; p++){
			for(int i=0; i<Nd; i++){
    			R1 = mt.rand(); R2 = mt.rand();

    			// Original PSO
                V[p][i] = w * V[p][i] + C1*R1*(pBestPosition[p][i] - R[p][i]) + C2*R2*(gBestPosition[i] - R[p][i]);
                if(V[p][i] > vMax) V[p][i] = vMin + mt.randDblExc(vMax-vMin);
                if(V[p][i] < vMin) V[p][i] = vMin + mt.randDblExc(vMax-vMin);
			}
		}
		timer.stopTimer();
		velocityTime += timer.getElapsedTime();
	} // End Time Steps

    timer1.stopTimer();
    totalTime += timer1.getElapsedTime();

    R.clear(); V.clear(); M.clear();
    pBestPosition.clear(); pBestValue.clear(); gBestPosition.clear();
        
	cout    << functionName << " "
            << gBestValue   << " " 
            << Np           << " "
            << Nd           << " "
            << lastStep     << " "
            << numEvals     << " "
            << positionTime << " "
            << fitnessTime  << " "
            << velocityTime << " "
            << totalTime << endl;
}

void runPSO(float xMin, float xMax, float vMin, float vMax, float (*rPtr)(ldArray2D& , int, int), string functionName){
    
    int Nt,numEvals;
    vector<int> Np(3, 0);
    vector<int> Nd(3, 0);
    
    Np[0] = 100; Np[1]=500; Np[2]=1000;
    Nd[0] = 100; Nd[1]=500; Nd[2]=1000;
    Nt = 10000;
    
    for(int i=0; i< Np.size(); i++){
        for(int j=0; j< Nd.size(); j++){
            for(int x=0; x<10; x++){
                PSO(Np[i], Nd[j], Nt, xMin, xMax, vMin, vMax, rPtr, numEvals, functionName);
            }
        }
    }
}
void run_SPSO_2007(int Np, float xMin, float xMax, float vMin, float vMax, float (*rPtr)(ldArray2D& , int, int), string functionName){

    int Nd, Nt, numEvals;
    int NdMin, NdMax, NdStep;
    
    NdMin = 50; NdMax = 100; NdStep = 10;
    Nt = 3000; Nd = 30;
    for(int x=0; x<10; x++){
        SPSO2007(Np, Nd, Nt, xMin, xMax, vMin, vMax, rPtr, numEvals, functionName);
    }
    cout << endl;
    
    for(Nd=NdMin, Np=200; Nd<=NdMax; Nd+=NdStep, Np+=40){
        for(int x=0; x<10; x++){
            SPSO2007(Np, Nd, Nt, xMin, xMax, vMin, vMax, rPtr, numEvals, functionName);
        }
        cout << endl;
    }

	cout << endl;
}
int main(){

    int Np;
	float (*rPtr)(ldArray2D& , int, int) = NULL;
	float xMin, xMax, vMin, vMax;
       
    cout << "Function, Fitness, Last Step, Np, Nd, Evals, Position Time, Fitness Time, Velocity Time, Total Time" << endl;
    
    rPtr = &F1; Np = 120;
    xMin = -100; xMax = 100;
    vMin = -100; vMax = 100;
    runPSO(xMin, xMax, vMin, vMax, rPtr, "F1");
    run_SPSO_2007(Np, xMin, xMax, vMin, vMax, rPtr, "F1");
   
    rPtr = NULL;
	return 0;
}
