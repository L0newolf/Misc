#include <iostream>
#include <fstream>
#include <stdio.h>      /* printf, fopen */
#include <stdlib.h>  


#include <unistd.h> // usleep, write
#include <sys/time.h> // gettimeofday
#include <pthread.h>
#include <fcntl.h> //file io
#include <sys/stat.h> //stat()
#include <errno.h>
#include <dirent.h> // accessing dir functions

#include<sys/socket.h>
#include<arpa/inet.h> //sockaddr_in, htons()
#include <signal.h>

#define TCP_SERVER_PORT_FOR_VIDEO 8899

void create_listen_TCP_for_video() {
    int socket_desc, new_socket_conn, size_sockaddr;
    struct sockaddr_in server, client;

    //Create socket
    socket_desc = socket(AF_INET , SOCK_STREAM , 0);
    if (socket_desc == -1) {
        perror("Could not create socket in create_listen_TCP_for_video");
        exit(-1);
    }
    printf("TCP Server socket created successfully in create_listen_TCP_for_video\n");

    //Prepare the sockaddr_in structure
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(TCP_SERVER_PORT_FOR_VIDEO);

    //Bind socket to TCP SERVER Port number to the BBB IP
    if (bind(socket_desc,(struct sockaddr *)&server , sizeof(server)) < 0) {
        perror("bind failed in create_listen_TCP_for_video");
        exit(-1);
    }
    printf("Binding done successfully in create_listen_TCP_for_video\n");

    //Listen with pending connection Q set to 10
    if (listen(socket_desc, 10)<0) {
        perror("listen failed in create_listen_TCP_for_video");
        exit(-1);
    }
    printf("socket set to listen mode to ack its willingness to accept incoming connections in create_listen_TCP_for_video\n\n");

    //block SIGPIPE signal - cause of 141 error
    sigset_t set;
    sigaddset(&set, SIGPIPE);
    int retcode = sigprocmask(SIG_BLOCK, &set, NULL);
    if (retcode == -1) perror("sigprocmask");

    //Start to accept any incoming connections
    printf("TCP server set and waiting for incoming connections in create_listen_TCP_for_video...\n");
    size_sockaddr = sizeof(struct sockaddr_in);

    // --------------------------------------------------------------------------------
    // looping forever to wait for any connections, actually only 1
    // --------------------------------------------------------------------------------
    while(1) {
      // note that it is blocked inside accept() - accept a new connection

      printf("Waiting for new connection .... \n");

      if ((new_socket_conn = accept(socket_desc, (struct sockaddr *)&client, (socklen_t*)&size_sockaddr))) {
        printf("Connection accepted\n");
        char *client_ip = inet_ntoa(client.sin_addr);
        int client_port = ntohs(client.sin_port);
        printf("Client PC IP = %s, Client PC Port = %d\n",client_ip,client_port);
      }
      else {
        perror("accept failed in create_listen_TCP_for_video");
        exit(-1);
      }

      // whether there's error or not inside, will close conn and return to accept
      //get_name_read_file_write_socket(new_socket_conn);

/*
      // READ file into buf and SEND it off in chunks of 1024 bytes
      // if err inside will send 4 byte word (fsize) 0 to indicate PC to skip the DL
      read_file_write_socket(FILE_IMG, new_socket_conn);

      // READ file into buf and SEND it off in chunks of 1024 bytes
      // if err inside will send 4 byte word (fsize) 0 to indicate PC to skip the DL
      read_file_write_socket(FILE_IMGTH, new_socket_conn);
*/

      if (close(new_socket_conn)<0) {
         perror("close:");
      }
    }
}

    // --------------------------------------------------------------------------------
    // Test Main
    // --------------------------------------------------------------------------------

int main()
{
  create_listen_TCP_for_video();
  return 0;
}


void get_name_read_file_write_socket(int socket_no)
{
    char fname[100];
    char get_name_cmd[10];
    char buf[1024];

    struct stat st;
    int fd, fsize;
    char *errmsg;
    int n,l,retval;

    //retval=read(socket_no, buf, 20); // yymmddhhmmss.avi (16 chars)
    //if (retval!=20) return;

    retval=read(socket_no, buf, 100); // get>yymmddhhmmss.avi (4+16 chars) or get>thyymmddhhmmss.avi (4+18 char length)
    if ((retval!=20) && (retval!=22)) return;

    strncpy(get_name_cmd,buf,4);
    get_name_cmd[4]=0;
    if (strcmp(get_name_cmd,"get>")!=0) {
      perror("Serious Error: get name cmd failed in get_name_read_file_write_socket");
      printf("cmd is %s\n",get_name_cmd);
      printf("TCP thread create_listen_TCP_for_video terminated\n"); 
     exit(-1);
    }

    memset(fname,0,100);
    strcpy(fname,VIDEO_FOLDER);
    l=strlen(VIDEO_FOLDER);
    if (buf[4]=='t')
       strncpy(&fname[l],&buf[4],18);
    else
       strncpy(&fname[l],&buf[4],16);

    printf("\nfname is %s\n\n",fname);

    // find file size
    if (stat(fname, &st) != 0) {
      perror("stat() failed in get_name_read_file_write_socket");
      fsize=0;
      write(socket_no, &fsize, sizeof(fsize)); //send 0 to tell PC to skip this DL
      return;
    }

    // check if size is 0
    fsize=st.st_size;
    if (fsize==0) {
      write(socket_no, &fsize, sizeof(fsize));
      return; //0 detetced, so that means tell PC to skip this DL
    }

    // file size >0, now tries to open the file
    fd=open(fname,O_RDONLY);
    if (fd==-1) {
      perror("open failed in get_name_read_file_write_socket");
      errmsg = strerror(errno);
      printf("%s: %s\n", fname, errmsg);

      fsize=0;
      write(socket_no, &fsize, sizeof(fsize)); //send 0 to tell PC to skip this DL
      return;
    }

    // so far ok
    // so write the size to tell PC to continue this DL
    // it writes size, follow by the file in chunks of 1024 bytes
    //
    retval=write(socket_no, &fsize, sizeof(fsize));
    if (retval!=sizeof(fsize)) {
      perror("write socket error for fsize:");
      return;
    }

    // write the file to socket
    while (1) {
      n=read(fd,buf,1024); // n is number of bytes read

      if (n==0) break; // EOF

      if (n<0) {
        perror("read failed in get_name_read_file_write_socket");
        printf("fd is %d\n",fd);
        return;
      }

      // Need a loop for the write, because not all of the data may be written
      // in one call; write will return how many bytes were written.
      // p ptr keeps track of where in the buffer we are, while we decrement n
      // to keep track of how many bytes are left to write.

      char *p = buf;
      while (n > 0) {
        int bytes_written = write(socket_no, p, n); // write to socket
        if (bytes_written <= 0) {
          perror("write AVI to sock receives -ve or 0 values"); //e.g, pipe broken error - conn closed on client side
          close(fd);
          return; //upper layer will close conn
        }
        n -= bytes_written;
        p += bytes_written;
      }

      //retval=write(socket_no, buf, n); //write to socket

      //if (retval<=0) {
      //  perror("write AVI to sock receives -ve or 0 values:"); //e.g, pipe broken error - conn closed on client side
      //  close(fd);
      //  return; //upper layer will close conn
      //}

      //if (retval<n) {
      //  printf("write AVI to sock: numbytes sent(%d)<n(%d), close sock and back to 'accept' \n",retval,n);
      //  perror("write:");
      //  close(fd);
      //  return;
      //}
    }

    // success
    close(fd);

    // delete the AVI file which has already been sent out
    n=remove(fname);
    if (n<0) {
      perror("remove failed in get_name_read_file_write_socket");
      printf("%s\n",fname);
      return;
    }
    printf("%s removed!\n",fname);
}