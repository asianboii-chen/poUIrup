SOURCE_DIR = ../source
LIBRARY_DIR = ../library

SOURCES = $(shell find $(SOURCE_DIR) -type f -name *.cc)

all: development

development: $(SOURCES)
	g++ -std=gnu++20 -I/usr/local/include -I/usr/local/include/FL/images -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_THREAD_SAFE -D_REENTRANT -pthread -lpthread -framework Cocoa $(LIBRARY_DIR)/*.a $^

.PHONY: all development
