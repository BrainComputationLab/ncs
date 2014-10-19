// Definition of the ClientSocket class

#ifndef ClientSocket_class
#define ClientSocket_class

#include <ncs/spec/Socket.h>
#include <ncs/spec/SocketException.h>

namespace ncs {

namespace spec {

class ClientSocket : private ncs::spec::Socket
{
 public:

  ClientSocket ( std::string host, int port );
  virtual ~ClientSocket(){};

  const ClientSocket& operator << ( const std::string& ) const;
  const ClientSocket& operator >> ( std::string& ) const;

};

} // namespace spec

} // namespace ncs


#endif
