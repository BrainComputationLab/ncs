#ifndef ClientSocket_class
#define ClientSocket_class

#include <ncs/spec/Socket.h>
#include <ncs/spec/SocketException.h>
#include <iostream>

namespace ncs {

namespace spec {

class ClientSocket : private ncs::spec::Socket
{
 public:
  ClientSocket ();
  ClientSocket (std::string host, int port);
  virtual ~ClientSocket(){};

  bool bindWithoutThrow (std::string host, int port);

  const ClientSocket& operator << (const std::string&) const;
  const ClientSocket& operator >> (std::string&) const;

  void close();
};

} // namespace spec

} // namespace ncs


#endif
