import qrcode

url = input("Enter Text or URL: ")

img = qrcode.make(url)

img.save('qrcode.png')

print('QR code Created successfully')