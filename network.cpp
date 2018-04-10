#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <math.h>
using namespace std;

#define e 2.718281828
#define ETA 1.52
#define hl 5

double sigmoid(double x){
	return 1/(double)(1+pow(e,-x));
}

double d_sigmoid(double x){
	return x*(1-x);
}
/*
NETWORK ASPECTS
64 i/p
64*5 w(0)
5-10 hidden
5*10 w(1);
10 o/p
sigmoid of linear combination at each level
cross entropy error fn
{-sum(all->n)sum(all->k)t<nk>ln(y<nk>)} will be used
*/
//below is 5*64
vector <vector <double> > w_0;//64 weights for each of 5 first layer nodes,between input and first layer
//below is 10*6
vector <vector <double> > w_1;//5 weights for each of 10 output nodes,between first layer and output

vector <vector <double> > data_points;/*last value of data point
 is class type(0-9)*/
//ABOVE 2D VECTOR IS 3100*65
vector <vector <double> > v_points;//points read from validation set

vector <vector <double> > test_points;//points read from testing set

/*
FUNCTION FORWARD
This will bypass points through the current neural
network and give out vectors into 'y',passed as a parameter
*/
vector <vector <double> > hidden_layer;//for every 100 pts,100*5
double del_e_1[10][hl+1]={0.0};//del K,10*6
double del_e_0[hl][64]={0.0};//del K,5*64
//BELOW STORES THE PREDICTION FOR EVERY 100 POINTS,BATCH SIZE
vector <vector <double> > y;//TRAINING USAGE

vector <vector <double> > predict_y;//store output of validation at every iteration,here.


void backward(int start_index);




void forward_prop(int start_index){
    //cout<<"entered forward!!\n\n\n.....\n";
    int i,j,k;
    double t;
    vector <double> temp;vector <double> temp2;
    for(i=0;i<100;i++){
        //cout<<"\t"<<i+start_index<<"\n";
        for(j=0;j<hl;j++){
            t=0;
            for(k=0;k<64;k++){
                t += data_points[start_index + i][k] * w_0[j][k];
                //cout<<"\t\t"<<t<<"\n";
            }
            temp.push_back(t);
        }//temp ----> 5,5 = hl
        for(j=0;j<temp.size();j++){
        	temp[j] = tanh(temp[j]);
        }
        hidden_layer.push_back(temp);//hidden layer(5 nodes) at each pt.
        //cout<<"hidden layer size:"<<hidden_layer.size()<<"\n";

        for(j=0;j<10;j++){
            t = 0;
            for(k=0;k<hl;k++){
                t += temp[k] * w_1[j][k];
            }
            t += w_1[j][hl];//BIAS
            //cout<<"\t\t\t\t"<<t<<"\n";
            temp2.push_back(t);
        }
        for(j=0;j<temp2.size();j++){
            temp2[j] = sigmoid(temp2[j]);
        }
        y.push_back(temp2);
        temp.clear();
        temp2.clear();
        //cout<<"\t current size of y:"<<y.size()<<"*"<<y[i].size()<<"\n";
    }
    //CALL BACKWARD FUNCTION TO MAKE CHANGES TO WEIGHTS
    backward(start_index);
}

void backward(int start_index){
	int i,j,k;
	vector <double> del_k;
	vector <double> del_j;
	for(i=0;i<100;i++){
		for(j=0;j<10;j++){
            del_k.push_back(y[i][j]);
		}
		del_k[ data_points[start_index+i][64] ]--;
		for(k=0;k<10;k++){
			for(j=0;j<hl;j++){
				del_e_1[k][j]+= del_k[k] * hidden_layer[i][j];

			}
			del_e_1[k][j] += del_k[k]*1.0;
		}
		double sum = 0.0;
		for(j=0;j<hl;j++){
			sum = 0.0;
			for(k=0;k<10;k++){
				sum += w_1[k][j]*del_k[k];
			}
			del_j.push_back(sum*d_sigmoid(hidden_layer[i][j]));
		}
		for(j=0;j<hl;j++){
			for(k=0;k<64;k++){
				del_e_0[j][k] += del_j[j]*data_points[start_index+i][k];
			}
		}

		del_j.clear();
		del_k.clear();
	}
	for(j=0;j<hl;j++){
			for(k=0;k<64;k++){
				del_e_0[j][k] /=100.0;
				w_0[j][k] -= ETA*del_e_0[j][k];

			}
		}
	for(j=0;j<10;j++){
			for(k=0;k<hl+1;k++){
				del_e_1[j][k] /=100.0;
				w_1[j][k] -= ETA*del_e_1[j][k];
			}
		}



	hidden_layer.clear();
	y.clear();
}

//Validate will give out the error of the current NN
double validate(){
    //cout<<"\t\t\t\t\t\tentered validate!!\n\n\n.....\n";
    int i,j,k;
    double t;
    double v_error = 0;
    vector <double> temp;vector <double> temp2;
    for(i=0;i<v_points.size();i++){
        //cout<<"\t"<<i+start_index<<"\n";
        for(j=0;j<hl;j++){
            t=0;
            for(k=0;k<64;k++){
                t += v_points[i][k] * w_0[j][k];
                //cout<<"\t\t"<<t<<"\n";
            }
            temp.push_back(t);
        }
        for(j=0;j<temp.size();j++){
        	temp[j] = tanh(temp[j]);
        }

        for(j=0;j<10;j++){
            t = 0;
            for(k=0;k<hl;k++){
                t += temp[k] + w_1[j][k];
            }
            t += w_1[j][hl];//BIAS
            //cout<<"\t\t\t\t"<<t<<"\n";
            temp2.push_back(t);
        }
        for(j=0;j<temp2.size();j++){
            temp2[j] = sigmoid(temp2[j]);


        }
        temp2[ v_points[i][64] ]--;
        for(j=0;j<temp2.size();j++){
            v_error += pow(temp2[j],2);
        }

        temp.clear();
        temp2.clear();

    }
    return v_error;
}

//OUTPUT IS Y(10 VALUES) FOR ALL TEST POINTS
void predict(){
    //cout<<"\t\t\t\t\t\tentered predict!!\n\n\n.....\n";
    int i,j,k;
    double t;
    double v_error = 0;
    vector <double> temp;vector <double> temp2;
    for(i=0;i<test_points.size();i++){
        //cout<<"\t"<<i+start_index<<"\n";
        for(j=0;j<hl;j++){
            t=0;
            for(k=0;k<64;k++){
                t += test_points[i][k] * w_0[j][k];
                //cout<<"\t\t"<<t<<"\n";
            }
            temp.push_back(t);
        }
        for(j=0;j<temp.size();j++){
        	temp[j] = tanh(temp[j]);
        }

        for(j=0;j<10;j++){
            t = 0;
            for(k=0;k<hl;k++){
                t += temp[k] + w_1[j][k];
            }
            t += w_1[j][hl];//BIAS
            //cout<<"\t\t\t\t"<<t<<"\n";
            temp2.push_back(t);
        }
        for(j=0;j<temp2.size();j++){
            temp2[j] = sigmoid(temp2[j]);

        }
        predict_y.push_back(temp2);


        temp.clear();
        temp2.clear();

    }

}





int main(){
	//TRAINING DATA FORMAT
	ifstream infile("/Users/vishnu/Desktop/ML2-Assignment2/train.txt");
	string line;
	double x;int i;
	vector <double> temp_point;
	int c;
	while(getline(infile,line)){
		stringstream ss(line);
		while(ss >> x){
			temp_point.push_back(x);
			if(ss.peek()==',')
				ss.ignore();
		}
		data_points.push_back(temp_point);
		temp_point.clear();

	}
    //VALIDATION DATA FORMAT
	ifstream infile2("/Users/vishnu/Desktop/ML2-Assignment2/validation.txt");
	string line2;
	vector <double> temp_point2;
	while(getline(infile2,line2)){
		stringstream ss2(line2);
		while(ss2 >> x){
			temp_point2.push_back(x);
			if(ss2.peek()==',')
				ss2.ignore();
		}
		v_points.push_back(temp_point2);
		temp_point2.clear();

	}
	//cout<<"Size of validation points...\n"<<v_points.size()<<"*"<<v_points[10].size()<<"\n\n\n"<<v_points[305][64]<<"\n";

    //TESTING DATA FORMAT
	ifstream infile3("/Users/vishnu/Desktop/ML2-Assignment2/test.txt");
	string line3;
	vector <double> temp_point3;
	while(getline(infile3,line3)){
		stringstream ss3(line3);
		while(ss3 >> x){
			temp_point3.push_back(x);
			if(ss3.peek()==',')
				ss3.ignore();
		}
		test_points.push_back(temp_point3);
		temp_point3.clear();

	}
	//cout<<"Size of testing points...\n"<<test_points.size()<<"*"<<test_points[10].size()<<"\n\n\n"<<test_points[150][64]<<"\n";


	//FILES PARSED
	//SET UP NETWORK,RANDOM WEIGHTS AT FIRST
	vector <double> temp_w;
	int j;
	for(i=0;i<hl;i++){
		for(j=0;j<64;j++){
			temp_w.push_back(-1 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(2))) );
		}
		w_0.push_back(temp_w);
		temp_w.clear();
	}
   // cout<<w_0[hl-1][64]<<"\n";
	for(i=0;i<10;i++){
		for(j=0;j<hl+1;j++){
			temp_w.push_back(-1 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(2))) );
		}
		w_1.push_back(temp_w);
		temp_w.clear();
	}
	//cout<<w_1[9][hl-1]<<"\n\n";
	//initiate forward & backward propagation
    //Have to do validation of the network at every iteraqtion
    double check;
    int stop_check = 0;
	for(i=0;i<data_points.size();i+=100){
		//if(stop_check == 3)break;
		cout<<"start index:"<<i<<"\n";
		forward_prop(i);//does forward and backward,updates weights

		//cout<<"random w_0 value:"<<w_0[hl-1][63]<<"\nrandom w_1 value:"<<w_1[9][hl]<<"\n\n";

        if(i==0){
            cout<<"BASE__________________CASE\n";
            check = validate();
        }
        if(i!=0){
            if( check<validate() ){
               	stop_check++;
               	//cout<<"TRAINING BACKWARD....\n";
                //printf("i: %d\n",i);
                //break;

            }
            if( check >= validate() ){
                check = validate();
                stop_check = 0;
            }
        }
	}
    double ans;
    int ind;
    vector<int>ans1;
    predict();
    cout<<"\n\nsize of test points:"<<test_points.size()<<"\n";
	for(int i = 0 ; i < test_points.size();i++){
        cout<<".\n";
        ans=0.0;
        for(int j = 0 ; j < predict_y[i].size(); j++){
            if(predict_y[i][j]>ans){
                ans=predict_y[i][j];
                ind=j;
            }
            //cout<<predict_y[i][j]<<" ";
        }
            //cout<<endl;
        ans1.push_back(ind);
	}
	double cnt1=0.0,cnt2=0.0;
	for(int i=0;i<test_points.size();i++){
        if(test_points[i][64]==ans1[i] )
            cnt1++;
        else
            cnt2++;
	}
	double accuracy =  (double)cnt1/(double)(cnt1 + cnt2);
    cout<<accuracy<<endl;


	return 0;
}
