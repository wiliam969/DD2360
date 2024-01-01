%%writefile exec.sh

for i in {100..10000..10}
do
  ./ex 128 $i >> errors.txt
done
