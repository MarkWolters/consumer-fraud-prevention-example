Consumer Fraud Prevention Example
=================================

NOTE - this example requires DSE Enterprise with Analytics enabled 

## Scenario

This is an example of how to use DSE Analytics to detect fraudulent claims in a real time environment.  
This example deals with a credit card processing application using Spark.MLlib to learn about fraudulent claims.
The application reads from a sample data set and then detect actual fraudulent claims from a separate data set containing simulated data.

## Schema Setup

1. Start DSE in Analytics mode
2. Run the included cql script resources/cql/create_schema.cql using cqlsh with the following command
	cqlsh <ip address> -f '<project directory>/resources/cql/create_schema.cql'

## Configuring the processor
1. Open the file src/main/resources/properties.txt
2. set trainingDir=<location of the training data> (located in src/main/resources/training)
3. set testDir=<location of test data> (located in src/main/resources/test)
4. set batchDuration and numFeatures to appropriate values.  Defaults are respectively 1 and 3

## TODO: above section needs more fleshing out

## Building and running the jar file - WORK IN PROGRESS: CURRENTLY BUILDS BUT DOES NOT RUN CORRECTLY

To build the jar file run:

	mvn clean package 

To start the processor run

	dse spark-submit <path to jar file> <path to properties.txt file>
	
