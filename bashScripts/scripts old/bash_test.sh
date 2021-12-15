 #!/bin/bash


for i in 0 5 10 50 100 500
do
	
	if [ $i -lt 100 ]
	then
		if [ $i -lt 10 ]
		then 
			i_text="00${i}"
		else
			i_text="0${i}"
		fi
	else
		i_text="${i}"
	fi
	printf "%s\n" $i_text
done	


# declare an array called array and define 3 vales
array=( "one" "two" "three" )
for i in ${array[@]}
do
	echo $i
done