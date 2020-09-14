#include "iostream"
#include "math.h"
#include "stdlib.h"
#include "time.h"
#include "vector"
#include "assert.h"
#include "string.h"
using namespace std;

#define innode  2       //Enter the number of nodes, and 2 addends will be entered
#define hidenode  26    //Hide the number of nodes, store "carrying bits"
#define outnode  1      //Output the number of nodes, a predicted number will be output
#define alpha  0.1      //Learning rate
#define binary_dim 8    //Maximum length of binary number

#define randval(high) ( (double)rand() / RAND_MAX * high )
#define uniform_plus_minus_one ( (double)( 2.0 * rand() ) / ((double)RAND_MAX + 1.0) - 1.0 )  //Uniform random distribution


int largest_number = ( pow(2, binary_dim) );  //The largest decimal number that can be represented corresponding to the maximum length of the binary system

//Activation function
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

//The derivative of the activation function, y is the activation function value
double dsigmoid(double y)
{
    return y * (1.0 - y);
}

//The derivative of tanh, y is the value of tanh
double dtanh(double y)
{
    y = tanh(y);
    return 1.0 - y * y;
}

//Convert a decimal integer to a binary number
void int2binary(int n, int *arr)
{
    int i = 0;
    while(n)
    {
        arr[i++] = n % 2;
        n /= 2;
    }
    while(i < binary_dim)
        arr[i++] = 0;
}

class RNN
{
public:
    RNN();
    virtual ~RNN();
    void train();

public:
    double W_I[innode][hidenode];     //Connect the input and the weight matrix of the input gate in the hidden layer unit
    double U_I[hidenode][hidenode];   //Connect the weight matrix of the output of the previous hidden layer and the input gate in the hidden layer unit
    double W_F[innode][hidenode];     //Connect the input and the weight matrix of the forget gate in the hidden layer unit
    double U_F[hidenode][hidenode];   //Connect the weight matrix of the forget door in the previous hidden layer and the hidden layer unit
    double W_O[innode][hidenode];     //Connect the input and the weight matrix of the forget gate in the hidden layer unit
    double U_O[hidenode][hidenode];   //The weight matrix connecting the previous hidden layer and the hidden layer at the present moment
    double W_G[innode][hidenode];     //Weight matrix used to generate new memory
    double U_G[hidenode][hidenode];   //Weight matrix used to generate new memory
    double W_out[hidenode][outnode];  //The weight matrix connecting the hidden layer and the output layer

    double *x;             //layer 0 output value, set directly by the input vector
    //double *layer_1; //layer 1 output value
    double *y;             //layer 2 output value
};

void winit(double w[], int n) //Weight initialization
{
    for(int i=0; i<n; i++)
        w[i] = uniform_plus_minus_one;  //Uniform random distribution
}

RNN::RNN()
{
    x = new double[innode];
    y = new double[outnode];
    winit((double*)W_I, innode * hidenode);
    winit((double*)U_I, hidenode * hidenode);
    winit((double*)W_F, innode * hidenode);
    winit((double*)U_F, hidenode * hidenode);
    winit((double*)W_O, innode * hidenode);
    winit((double*)U_O, hidenode * hidenode);
    winit((double*)W_G, innode * hidenode);
    winit((double*)U_G, hidenode * hidenode);
    winit((double*)W_out, hidenode * outnode);
}

RNN::~RNN()
{
    delete x;
    delete y;
}

void RNN::train()
{
    int epoch, i, j, k, m, p;
    vector<double*> I_vector;      //Enter the gate
    vector<double*> F_vector;      //The Forgotten Door
    vector<double*> O_vector;      //Output gate
    vector<double*> G_vector;      //New memory
    vector<double*> S_vector;      //Status value
    vector<double*> h_vector;      //output value
    vector<double> y_delta;        //Save the partial derivative of the error about the output layer

    for(epoch=0; epoch<11000; epoch++)  //Training times
    {
        double e = 0.0;  //error

        int predict[binary_dim];               //Save the predicted value generated each time
        memset(predict, 0, sizeof(predict));

        int a_int = (int)randval(largest_number/2.0);  // randomly generate an addend a
        int a[binary_dim];
        int2binary(a_int, a);                 //Convert to binary number

        int b_int = (int)randval(largest_number/2.0);  // randomly generate another addend b
        int b[binary_dim];
        int2binary(b_int, b);                 //Convert to binary number

        int c_int = a_int + b_int;            //True and c
        int c[binary_dim];
        int2binary(c_int, c);                 //Convert to binary number

        //There is no previous hidden layer at time 0, so initialize a all 0
        double *S = new double[hidenode];     //Status value
        double *h = new double[hidenode];     //output value

        for(i=0; i<hidenode; i++)
        {
            S[i] = 0;
            h[i] = 0;
        }
        S_vector.push_back(S);
        h_vector.push_back(h);

        //Forward propagation
        for(p=0; p<binary_dim; p++)           //Loop through the binary array, starting from the lowest bit
        {
            x[0] = a[p];
            x[1] = b[p];
            double t = (double)c[p];          //Actual value
            double *in_gate = new double[hidenode];     //Enter the gate
            double *out_gate = new double[hidenode];    //Output gate
            double *forget_gate = new double[hidenode]; //The Forgotten Door
            double *g_gate = new double[hidenode];      //New memory
            double *state = new double[hidenode];       //Status value
            double *h = new double[hidenode];           //Hidden layer output value

            for(j=0; j<hidenode; j++)
            {
                //The input layer is broadcast to the hidden layer
                double inGate = 0.0;
                double outGate = 0.0;
                double forgetGate = 0.0;
                double gGate = 0.0;
                double s = 0.0;

                for(m=0; m<innode; m++)
                {
                    inGate += x[m] * W_I[m][j];
                    outGate += x[m] * W_O[m][j];
                    forgetGate += x[m] * W_F[m][j];
                    gGate += x[m] * W_G[m][j];
                }

                double *h_pre = h_vector.back();
                double *state_pre = S_vector.back();
                for(m=0; m<hidenode; m++)
                {
                    inGate += h_pre[m] * U_I[m][j];
                    outGate += h_pre[m] * U_O[m][j];
                    forgetGate += h_pre[m] * U_F[m][j];
                    gGate += h_pre[m] * U_G[m][j];
                }

                in_gate[j] = sigmoid(inGate);
                out_gate[j] = sigmoid(outGate);
                forget_gate[j] = sigmoid(forgetGate);
                g_gate[j] = sigmoid(gGate);

                double s_pre = state_pre[j];
                state[j] = forget_gate[j] * s_pre + g_gate[j] * in_gate[j];
                h[j] = in_gate[j] * tanh(state[j]);
            }


            for(k=0; k<outnode; k++)
            {
                //The hidden layer propagates to the output layer
                double out = 0.0;
                for(j=0; j<hidenode; j++)
                    out += h[j] * W_out[j][k];
                y[k] = sigmoid(out);               //The output of each unit of the output layer
            }

            predict[p] = (int)floor(y[0] + 0.5);   //Record predicted value

            //Save the hidden layer for next calculation
            I_vector.push_back(in_gate);
            F_vector.push_back(forget_gate);
            O_vector.push_back(out_gate);
            S_vector.push_back(state);
            G_vector.push_back(g_gate);
            h_vector.push_back(h);

            //Save the partial derivative of the standard error with respect to the output layer
            y_delta.push_back( (t - y[0]) * dsigmoid(y[0]) );
            e += fabs(t - y[0]);          //error
        }

        //Error back propagation

        //Hidden layer deviation, calculated by the hidden layer error at a later point in time and the current output layer error
        double h_delta[hidenode];
        double *O_delta = new double[hidenode];
        double *I_delta = new double[hidenode];
        double *F_delta = new double[hidenode];
        double *G_delta = new double[hidenode];
        double *state_delta = new double[hidenode];
        //A hidden layer error after the current time
        double *O_future_delta = new double[hidenode];
        double *I_future_delta = new double[hidenode];
        double *F_future_delta = new double[hidenode];
        double *G_future_delta = new double[hidenode];
        double *state_future_delta = new double[hidenode];
        double *forget_gate_future = new double[hidenode];
        for(j=0; j<hidenode; j++)
        {
            O_future_delta[j] = 0;
            I_future_delta[j] = 0;
            F_future_delta[j] = 0;
            G_future_delta[j] = 0;
            state_future_delta[j] = 0;
            forget_gate_future[j] = 0;
        }
        for(p=binary_dim-1; p>=0 ; p--)
        {
            x[0] = a[p];
            x[1] = b[p];

            //Current hidden layer
            double *in_gate = I_vector[p];     //Enter the gate
            double *out_gate = O_vector[p];    //Output gate
            double *forget_gate = F_vector[p]; //The Forgotten Door
            double *g_gate = G_vector[p];      //New memory
            double *state = S_vector[p+1];     //Status value
            double *h = h_vector[p+1];         //Hidden layer output value

            //Previous hidden layer
            double *h_pre = h_vector[p];
            double *state_pre = S_vector[p];

            for(k=0; k<outnode; k++)  //For each output unit in the network, update the weight
            {
                //Update the connection rights between the hidden layer and the output layer
                for(j=0; j<hidenode; j++)
                    W_out[j][k] += alpha * y_delta[p] * h[j];
            }

            //For each hidden unit in the network, calculate the error term and update the weight
            for(j=0; j<hidenode; j++)
            {
                h_delta[j] = 0.0;
                for(k=0; k<outnode; k++)
                {
                    h_delta[j] += y_delta[p] * W_out[j][k];
                }
                for(k=0; k<hidenode; k++)
                {
                    h_delta[j] += I_future_delta[k] * U_I[j][k];
                    h_delta[j] += F_future_delta[k] * U_F[j][k];
                    h_delta[j] += O_future_delta[k] * U_O[j][k];
                    h_delta[j] += G_future_delta[k] * U_G[j][k];
                }

                O_delta[j] = 0.0;
                I_delta[j] = 0.0;
                F_delta[j] = 0.0;
                G_delta[j] = 0.0;
                state_delta[j] = 0.0;

                //The correction error of the hidden layer
                O_delta[j] = h_delta[j] * tanh(state[j]) * dsigmoid(out_gate[j]);
                state_delta[j] = h_delta[j] * out_gate[j] * dtanh(state[j]) +
                                 state_future_delta[j] * forget_gate_future[j];
                F_delta[j] = state_delta[j] * state_pre[j] * dsigmoid(forget_gate[j]);
                I_delta[j] = state_delta[j] * g_gate[j] * dsigmoid(in_gate[j]);
                G_delta[j] = state_delta[j] * in_gate[j] * dsigmoid(g_gate[j]);

                //Update the weight between the previous hidden layer and the current hidden layer
                for(k=0; k<hidenode; k++)
                {
                    U_I[k][j] += alpha * I_delta[j] * h_pre[k];
                    U_F[k][j] += alpha * F_delta[j] * h_pre[k];
                    U_O[k][j] += alpha * O_delta[j] * h_pre[k];
                    U_G[k][j] += alpha * G_delta[j] * h_pre[k];
                }

                //Update the connection rights between the input layer and the hidden layer
                for(k=0; k<innode; k++)
                {
                    W_I[k][j] += alpha * I_delta[j] * x[k];
                    W_F[k][j] += alpha * F_delta[j] * x[k];
                    W_O[k][j] += alpha * O_delta[j] * x[k];
                    W_G[k][j] += alpha * G_delta[j] * x[k];
                }

            }

            if(p == binary_dim-1)
            {
                delete  O_future_delta;
                delete  F_future_delta;
                delete  I_future_delta;
                delete  G_future_delta;
                delete  state_future_delta;
                delete  forget_gate_future;
            }

            O_future_delta = O_delta;
            F_future_delta = F_delta;
            I_future_delta = I_delta;
            G_future_delta = G_delta;
            state_future_delta = state_delta;
            forget_gate_future = forget_gate;
        }
        delete  O_future_delta;
        delete  F_future_delta;
        delete  I_future_delta;
        delete  G_future_delta;
        delete  state_future_delta;

        if(epoch % 1000 == 0)
        {
            cout << "error：" << e << endl;
            cout << "pred：" ;
            for(k=binary_dim-1; k>=0; k--)
                cout << predict[k];
            cout << endl;

            cout << "true：" ;
            for(k=binary_dim-1; k>=0; k--)
                cout << c[k];
            cout << endl;

            int out = 0;
            for(k=binary_dim-1; k>=0; k--)
                out += predict[k] * pow(2, k);
            cout << a_int << " + " << b_int << " = " << out << endl << endl;
        }

        for(i=0; i<I_vector.size(); i++)
            delete I_vector[i];
        for(i=0; i<F_vector.size(); i++)
            delete F_vector[i];
        for(i=0; i<O_vector.size(); i++)
            delete O_vector[i];
        for(i=0; i<G_vector.size(); i++)
            delete G_vector[i];
        for(i=0; i<S_vector.size(); i++)
            delete S_vector[i];
        for(i=0; i<h_vector.size(); i++)
            delete h_vector[i];

        I_vector.clear();
        F_vector.clear();
        O_vector.clear();
        G_vector.clear();
        S_vector.clear();
        h_vector.clear();
        y_delta.clear();
    }
}

int main()
{
    srand(time(NULL));
    RNN rnn;
    rnn.train();
    return 0;
}
