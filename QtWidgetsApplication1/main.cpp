#include "QtWidgetsApplication1.h"  
#include <QtWidgets/QApplication>  
#include <QTextCodec>   

int main(int argc, char *argv[])  
{  
    // QTextCodec::setCodecForCStrings is deprecated and removed in Qt 5.  
    // Use QTextStream or QString::fromUtf8 for encoding conversions.  
    // The following lines are updated to ensure compatibility with modern Qt versions.  

    QTextCodec *codec = QTextCodec::codecForName("UTF-8");  
    if (codec) {  
        QTextCodec::setCodecForLocale(codec); // Set codec for locale  
    }  

    QApplication a(argc, argv);  
    MainWindow w;  
    w.show();  
    return a.exec();  
}
