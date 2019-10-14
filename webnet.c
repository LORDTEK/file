#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#define theFunction 3// 0:omerAct 1:Sigmoid 2:TanH 3:Bent 4:SoftPlus 5:LReLU 6:Sinc
#define learningRate 0.1
#ifdef _WIN32
#include <windows.h>
#pragma warning (disable : 4996)
#define _CRT_SECURE_NO_WARNINGS
#define inputFile "C:\\Vision\\webNetINPUT.dat"
#define outputFile "C:\\Vision\\webNetOUTPUT.dat"
#define testFile "C:\\Vision\\webNetTEST.dat"
#elif __linux__
#define inputFile "/home/xq/webNetINPUT.dat"
#define outputFile "/home/xq/webNetOUTPUT.dat"
#define testFile "/home/xq/webNetTEST.dat"
#else
#define inputFile "/Users/Shared/CommonAll/webNetINPUT.dat"
#define outputFile "/Users/Shared/CommonAll/webNetOUTPUT.dat"
#define testFile "/Users/Shared/CommonAll/webNetTEST.dat"
#endif

double Activate(double value);
double Derivate(double value);

int main(int argc, char **argv, char **envp) {

    clock_t startTime, endTime;
    double cpu_time_used;
    startTime = clock();

    int loopNumber = 1, loopNumber_temp = 1, multimillion = 1000000;
    if(argv[1] != '\0'){// loopNumber can define by (first) argument
        loopNumber_temp = atof(argv[1]);
        if ( (loopNumber_temp > 0) && (loopNumber_temp < 1000000000) ){
            loopNumber = loopNumber_temp;
        }
    }
    int perIN=1, perOUT=1;
    int lineCounter=1, i=0, j=0, k=0, m=0, usedLine=0;

    // How many lines in file + numbers of input and output
    FILE *inputNumber = fopen(inputFile, "r+");
    char mychar;
    while(1){
        mychar = fgetc(inputNumber);
        if(lineCounter == 1){//input
            if(mychar == ' '){
                perIN++;
            }
        }
        else if(lineCounter == 2){//output
            if(mychar == ' '){
                perOUT++;
            }
        }
        if(mychar == EOF || (int)(mychar) == -1){
            break;
        }
        else if((int)(mychar) == 13 || (int)(mychar) == 10){
            lineCounter++;
        }
    }
    fclose(inputNumber);

    int trainerSetNumber = (int) (lineCounter/2);
    double *trainerIN = (double *) calloc(trainerSetNumber * perIN, sizeof(double));
    double *testIN = (double *) calloc(trainerSetNumber * perIN, sizeof(double));
    double *trainerOUT = (double *) calloc(trainerSetNumber * perOUT, sizeof(double));
    int perH = perIN + perOUT;
    int layerH = (int) ( (sqrt(pow(perIN, 2)+pow(perOUT, 2)) + perH)/8.0 + 2.0 );
    double *nodeIN = (double *) calloc(perIN + layerH*perH + perOUT, sizeof(double));
    double *errorIN = (double *) calloc(perIN + layerH*perH + perOUT, sizeof(double));
    double *nodeOUT = (double *) calloc(perIN + layerH*perH + perOUT, sizeof(double));
    double *errorOUT = (double *) calloc(perIN + layerH*perH + perOUT, sizeof(double));
    int numberWeight = perH * (perIN + perOUT + perH*(layerH-1));
    double *weight = (double *) calloc(numberWeight, sizeof(double));
    double *weightError = (double *) calloc(numberWeight, sizeof(double));

    // Get TEST data (input only)
    FILE *testData = fopen(testFile, "r+");
    lineCounter = 0;
    i = 0;
    #ifdef _WIN32
        char *lineFileTest = new char[1024];
        double *dataLineTest = new double[32];
        if( fgets(lineFileTest, 1024, testData) != NULL ) {
    #else
        char *lineFileTest = 0;
        size_t lengthLineTest = 0;
        if( (getline(&lineFileTest, &lengthLineTest, testData)) != -1 ) {
    #endif
            char *token = strtok(lineFileTest, " ");
            while (token != NULL) {
            *(testIN+i) = atof(token);
            i++;
            token = strtok(NULL, " ");
        }
    }
    fclose(testData);
    if(lineFileTest){
        free(lineFileTest);
    }

    // If exisit then get weight data if not create randomly
    FILE *existData = fopen(outputFile, "r");
    if(!existData){
        for(i=0; i<numberWeight; i++){
            srand(time(NULL)+i);
            *(weight+i) = ((rand()%10000)/10000.0)/(double)perH;
        }
    }
    else{
        i = 0;
#ifdef _WIN32
        char *lineFileExist = new char[1024];
        double *dataLineExist = new double[32];
        if( fgets(lineFileExist, 1024, existData) != NULL ) {
#else
        char *lineFileExist = 0;
        size_t lengthLineExist = 0;
        if( (getline(&lineFileExist, &lengthLineExist, existData)) != -1 ) {
#endif
            char *token = strtok(lineFileExist, " ");
            while (token != NULL) {
                if((atoi(token)==perIN) && i==0){
                    i++;
                    token = strtok(NULL, " ");
                }
                else if((atoi(token)==perOUT) && i==1){
                    i++;
                    token = strtok(NULL, " ");
                }
                else if(i>=2 && i<=401){
                    *(weight+i-2) = atof(token);
                    i++;
                    token = strtok(NULL, " ");
                }
            }
        }
        fclose(existData);
        if(lineFileExist){
            free(lineFileExist);
        }
    }

    // Get training data (input and output)
    FILE *inputData = fopen(inputFile, "r+");
    lineCounter = 0;
    i = 0;
#ifdef _WIN32
    char *lineFile = new char[1024];
    double *dataLine = new double[32];
    while( fgets(lineFile, 1024, inputData) != NULL ) {
#else
    char *lineFile = 0;
    size_t lengthLine = 0;
    while( (getline(&lineFile, &lengthLine, inputData)) != -1 ) {
#endif
        char *token = strtok(lineFile, " ");
        while (token != NULL) {
            if((lineCounter&1)==0?1:0){//input
                *(trainerIN+i) = atof(token);
                i++;
            }
            else{//output
                *(trainerOUT+j) = atof(token);//Must be between 0 and 1 (scale it down to this)
                j++;
            }
            token = strtok(NULL, " ");
        }
        lineCounter++;
    }
    fclose(inputData);
    if(lineFile){
        free(lineFile);
    }

    lineCounter = 0;
    for(i=0; i<loopNumber; /*i++*/){
        for(j=0; j<perIN + layerH*perH + perOUT; j++){
            *(nodeIN+j) = 0.0;
            *(errorOUT+j) = 0.0;
        }
        //FEED FORWARD
        for(j=0; j<perIN; j++){//INPUT ACTIVATION
            *(nodeIN+j) = *(trainerIN+j+lineCounter*perIN);
            *(nodeOUT+j) = *(nodeIN+j);//Do not activate (exception)
        }
        for(j=0; j<perIN; j++){//INPUT TO FIRST HIDDEN
            for(k=0; k<perH; k++){
                *(nodeIN+perIN+k) += *(weight + usedLine++) * *(nodeOUT+j);
                *(nodeOUT+perIN+k) = Activate(*(nodeIN+perIN+k));
            }
        }
        for(j=1; j<layerH; j++){//HIDDENS
            for(k=0; k<perH; k++){
                for(m=0; m<perH; m++){
                    *(nodeIN+perIN+j*perH+m) += *(weight + usedLine++) * *(nodeOUT+perIN+(j-1)*perH+k);
                    *(nodeOUT+perIN+j*perH+m) = Activate(*(nodeIN+perIN+j*perH+m));
                }
            }
        }
        for(j=0; j<perH; j++){//LAST HIDDEN TO OUTPUT
            for(k=0; k<perOUT; k++){
                *(nodeIN+perIN+perH*layerH+k) += *(weight + usedLine++) * *(nodeOUT+perIN+(layerH-1)*perH+j);
                *(nodeOUT+perIN+perH*layerH+k) = Activate(*(nodeIN+perIN+perH*layerH+k));
            }
        }

        //BACKPROPAGATION
        for(j=perOUT-1; j>=0; j--){//OUTPUT DERIVATION
            *(errorOUT+perIN+perH*layerH+j) = *(nodeOUT+perIN+perH*layerH+j) - *(trainerOUT+j+lineCounter*perOUT);
            *(errorIN+perIN+perH*layerH+j) = *(errorOUT+perIN+perH*layerH+j) * Derivate(*(nodeIN+perIN+perH*layerH+j));
        }
        for(j=perH-1; j>=0; j--){//LAST HIDDEN FROM OUTPUT
            for(k=perOUT-1; k>=0; k--){
                *(errorOUT+perIN+perH*(layerH-1)+j) += *(errorIN+perIN+perH*layerH+k) * *(weight + --usedLine);
                *(errorIN+perIN+perH*(layerH-1)+j) = *(errorOUT+perIN+perH*(layerH-1)+j) * Derivate(*(nodeIN+perIN+perH*(layerH-1)+j));
                *(weightError + usedLine) = *(errorIN+perIN+perH*layerH+k) * *(nodeOUT+perIN+perH*(layerH-1)+j);
                *(weight + usedLine) = *(weight + usedLine) - *(weightError + usedLine);
            }
        }
        for(j=layerH-2; j>=0; j--){//HIDDENS
            for(k=perH-1; k>=0; k--){
                for(m=perH-1; m>=0; m--){
                    *(errorOUT+perIN+j*perH+m) += *(errorIN+perIN+(j+1)*layerH+m) * *(weight + --usedLine);
                    *(errorIN+perIN+j*perH+m) = *(errorOUT+perIN+j*perH+m) * Derivate(*(nodeIN+perIN+j*perH+m));
                    *(weightError + usedLine) = *(errorIN+perIN+(j+1)*layerH+m) * *(nodeOUT+perIN+j*perH+m);
                    *(weight + usedLine) = *(weight + usedLine) - *(weightError + usedLine);
                }
            }
        }
        for(j=perIN-1; j>=0; j--){//INPUT FROM FIRST HIDDEN
            for(k=perH-1; k>=0; k--){
                *(errorOUT+j) += *(errorIN+perIN+k) * *(weight + --usedLine);
                *(errorIN+j) = *(errorOUT+j) * Derivate(*(nodeIN+j));
                *(weightError + usedLine) = *(errorIN+perIN+k) * *(nodeOUT+j);
                *(weight + usedLine) = *(weight + usedLine) - *(weightError + usedLine);
            }
        }
        for(j=0; j<numberWeight; j++){//weight check
            if( *(weight+j) > 1.0 ){//max check
                *(weight+j) = 1.0;
            }
            if( *(weight+j) < -1.0 ){//min check
                *(weight+j) = -1.0;
            }
        }
        lineCounter++;
        if(lineCounter == trainerSetNumber){
            lineCounter = 0;
        }

        // to repeat million times
        if(multimillion >= 1){
            multimillion = multimillion - 1;
        }
        else {
            multimillion = 1000000;
            i = i + 1;
        }
    }

    for(i=0, j=1; i<numberWeight; i++){//NaN check
        if(*(weight+i) != *(weight+i)){
            j=0;
            break;
        }
    }
    if(j){
        FILE *outputData = fopen(outputFile, "w+");
            fprintf(outputData, "%d", perIN);
            fprintf(outputData, " %d", perOUT);
            for(i=0; i<numberWeight; i++){
                fprintf(outputData, " %f", *(weight+i));
            }
        fclose(outputData);
    }

    for(i=0; i<(perIN+perH*layerH+perOUT); i++){//Clear nodeIN because of +=
        *(nodeIN+i) = 0.0;
    }
    //TEST
    for(i=0; i<perIN; i++){//INPUT ACTIVATION
        *(nodeIN+i) = *(testIN+i);
        *(nodeOUT+i) = *(nodeIN+i);//Do not activate (exception)
    }
    for(i=0; i<perIN; i++){//INPUT TO FIRST HIDDEN
        for(j=0; j<perH; j++){
            *(nodeIN+perIN+j) += *(weight + usedLine++) * *(nodeOUT+i);
            *(nodeOUT+perIN+j) = Activate(*(nodeIN+perIN+j));
        }
    }
    for(i=1; i<layerH; i++){//HIDDENS
        for(j=0; j<perH; j++){
            for(k=0; k<perH; k++){
                *(nodeIN+perIN+i*perH+k) += *(weight + usedLine++) * *(nodeOUT+perIN+(i-1)*perH+j);
                *(nodeOUT+perIN+i*perH+k) = Activate(*(nodeIN+perIN+i*perH+k));
            }
        }
    }
    for(i=0; i<perH; i++){//LAST HIDDEN TO OUTPUT
        for(j=0; j<perOUT; j++){
            *(nodeIN+perIN+perH*layerH+j) += *(weight + usedLine++) * *(nodeOUT+perIN+(layerH-1)*perH+i);
            *(nodeOUT+perIN+perH*layerH+j) = Activate(*(nodeIN+perIN+perH*layerH+j));
        }
    }
    for(i=0; i<perOUT; i++){
        printf("%f        ", *(nodeOUT+perIN+perH*layerH+i));
    }
    printf("\n");

    endTime = clock();
    cpu_time_used = ((double) (endTime - startTime)) / CLOCKS_PER_SEC;
    printf("Processing time = %f seconds\n", cpu_time_used);

    return (EXIT_SUCCESS);
}

double Activate(double value){
    double result = 0.0;
    if(theFunction==0){//omerAct
        if(value<-1.0){
            result = -1.0;
        }
        else{
            result = 1.0;
        }
        result = result*pow(fabs(value+1.0), 5.0/7.0);
    }
    else if(theFunction==1){//Sigmoid
        result = 1.0 / (1 + (pow(M_E, -1.0*value)));
    }
    else if(theFunction==2){//TanH
        result = (pow(M_E, value)-pow(M_E, -1.0*value))/(pow(M_E, value)+pow(M_E, -1.0*value));
    }
    else if(theFunction==3){//Bent
        result = value+(sqrt(value*value+1)-1)/2;
    }
    else if(theFunction==4){//SoftPlus
        result = log(1+pow(M_E, value));
    }
    else if(theFunction==5){//LReLU
        if(value<0.0){
            result = 0.01 * value;
        }
        else if(value>=0.0){
            result = value;
        }
    }
    else if(theFunction==6){//Sinc
        if(value==0.0){
            result = 1.0;
        }
        else{
            result = sin(value)/value;
        }
    }
    return result;
}

double Derivate(double value){
    double result = 0.0;
    if(theFunction==0){//omerAct
        if((value>-1.000001) && (value<-0.999999)){
            result = 37.0;
        }
        else{
            result = 5.0 / (7.0 * pow(fabs(value+1.0), 2.0/7.0));
        }
    }
    else if(theFunction==1){//Sigmoid
        result = Activate(value) * (1 - Activate(value));
    }
    else if(theFunction==2){//TanH
        result = 1.0 - (Activate(value) * Activate(value));
    }
    else if(theFunction==3){//Bent
        result = 1.0 + value/(2*sqrt(value*value+1));
    }
    else if(theFunction==4){//SoftPlus
        result = 1.0 / (1.0 + pow(M_E, -1.0*value));
    }
    else if(theFunction==5){//LReLU
        if(value<0.0){
            result = 0.01;
        }
        else if(value>=0.0){
            result = 1.0;
        }
    }
    else if(theFunction==6){//Sinc
        if(value==0.0){
            result = 0.0;
        }
        else{
            result = (cos(value)/value)-(sin(value)/(value*value));
        }
    }
    return result;
}
