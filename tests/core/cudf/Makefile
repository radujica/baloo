all: udf_c

udf_c:
	clang++-6.0 -shared -fPIC -w udf_c.cpp -o udf_c.so

clean:
	rm -rf udf_c.so
