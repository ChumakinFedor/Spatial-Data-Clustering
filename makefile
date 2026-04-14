CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm
TARGET = kmeans_app

# Цели, которые не являются файлами
.PHONY: all generate run plot clean full

all: $(TARGET)

$(TARGET): main.c kmeans.c kmeans.h
	$(CC) $(CFLAGS) -o $(TARGET) main.c kmeans.c $(LDFLAGS)

generate:
	python3 generate_data.py

run: $(TARGET)
	./$(TARGET)

plot:
	python3 plot_results.py

clean:
	rm -f $(TARGET) data.csv results.csv clusters.png

# Последовательный запуск всех шагов
full:
	$(MAKE) clean
	$(MAKE) generate
	$(MAKE) run
	$(MAKE) plot